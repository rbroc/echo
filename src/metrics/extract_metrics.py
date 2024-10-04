"""
Script to extract metrics for human and AI-generated datasets. 
See metrics/README.md for instructions on how to run.
"""

import multiprocessing as mp
import pathlib
import sys
from argparse import ArgumentParser

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.get_metrics import get_all_metrics, get_information_metrics

MODELS = ["beluga7b", "llama2_chat7b", "llama2_chat13b", "mistral7b"]
PROMPT_NUMBERS = [21]
DATASETS = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]


def get_ai_paths(
    ai_dir: pathlib.Path, dataset: str = "dailydialog", temp: float | int = 1
) -> list[pathlib.Path]:
    """
    Get all paths pertaining to a particular dataset (e.g., mrpc, dailymail_cnn, stories, dailydialog, etc.) for models
    """
    ai_paths = []

    if temp and not isinstance(temp, (int, float)):
        raise ValueError(f"Temperature must be a int or float, not {type(temp)}")
    if PROMPT_NUMBERS and not isinstance(PROMPT_NUMBERS, list):
        raise ValueError(f"Prompt number must be an list, not {type(PROMPT_NUMBERS)}")

    for model_name in MODELS:
        model_path = ai_dir / model_name

        for prompt_number in PROMPT_NUMBERS:
            file_identifier = f"{dataset}_prompt_{prompt_number}_temp{temp}.ndjson"
            file_path = model_path / file_identifier
            ai_paths.extend([file_path])

    if len(ai_paths) == 0:
        print(
            f"[WARNING:] Length of ai paths is zero. Ensure that you have valid arguments."
        )

    return ai_paths


def get_ai_metrics(
    ai_dir=pathlib.Path(__file__).parents[2]
    / "datasets_files"
    / "text"
    / "ai_datasets"
    / "clean_data",
    dataset: str = "mrpc",
    temp: int | float = 1,
    batch_size: int = 1,
    n_process: int = 1,
    compute_perplexity: bool = True,
    save_dir=None,
):
    """
    Extract metrics for AI completions

    Args:
        ai_dir: path to directory with AI completions
        dataset: name of dataset to extract metrics for
        temp: temperature of completions
        batch_size: batch size for processing
        n_process: number of processes to use for multiprocessing
        compute_perplexity: whether to compute perplexity and entropy manually with GPT-2 (True) or relying on textdescriptives (False)
        save_dir: path to save directory. If None, does not save.

    Returns:
        completions_df: dataframe with metrics for completions
    """
    # get paths, only for prompt_numbers 21 (as they are the 2.0 prompts that we settled on, but fn is capable of loading whatever you want!)
    ai_paths = get_ai_paths(ai_dir=ai_dir, dataset=dataset, temp=temp)
    ai_dfs = [pd.read_json(ai_path, lines=True) for ai_path in ai_paths]

    # concat
    ai_df = pd.concat(ai_dfs, ignore_index=True, axis=0)

    # drop doc length (as metrics adds it, and will get confused when it has two cols that are duplicate)
    ai_df = ai_df.drop(columns=["doc_length"])

    # extract metrics
    completions_df = get_all_metrics(
        ai_df, text_column="completions", batch_size=batch_size, n_process=n_process
    )

    if compute_perplexity:
        completions_df = get_information_metrics(
            completions_df,
            text_column="completions",
            model_id="gpt2",
            batch_size=batch_size,
        )

    # drop cols
    completions_df = completions_df.drop(columns=["completions", "prompt"])

    # mv model col to front if present in df
    if "model" in completions_df.columns:
        completions_df.insert(
            loc=1, column="model", value=completions_df.pop("model")
        )  # insert mdl col on 2nd position in df

    if save_dir:
        completions_df.to_csv(save_dir / f"{dataset}_completions_temp{temp}.csv")

    return completions_df


def get_human_metrics(
    human_dir,
    dataset: str = "mrpc",
    batch_size: int = 1,
    n_process: int = 1,
    compute_perplexity: bool = True,
    save_dir=None,
):
    """
    extract metrics for human data

    Args:
        human_dir: path to directory with human data
        dataset: name of dataset to extract metrics for
        batch_size: batch size for processing
        n_process: number of processes to use for multiprocessing
        compute_perplexity: whether to compute perplexity and entropy manually with GPT-2 (True) or relying on textdescriptives (False)
        save_dir: path to save directory. If None, does not save.

    Returns:
        source_df: dataframe with metrics for source text
        completions_df: dataframe with metrics for completions
    """
    # def paths
    human_path = human_dir / dataset / "data.ndjson"

    # load df
    human_df = pd.read_json(human_path, lines=True)

    # add model col
    human_df["model"] = "human"

    # process source, completions
    source_df = get_all_metrics(
        human_df, text_column="source", batch_size=batch_size, n_process=n_process
    )
    completions_df = get_all_metrics(
        human_df,
        text_column="human_completions",
        batch_size=batch_size,
        n_process=n_process,
    )

    # compute perplexity and entropy manually
    if compute_perplexity:
        model_id = "gpt2"

        print(f"[INFO:] Computing perplexity and entropy via {model_id} ...")
        source_df = get_information_metrics(
            source_df, text_column="source", model_id=model_id, batch_size=batch_size
        )
        completions_df = get_information_metrics(
            completions_df,
            text_column="human_completions",
            model_id="gpt2",
            batch_size=batch_size,
        )

    # format dfs
    cols_to_drop = ["source", "human_completions"]

    for df in [source_df, completions_df]:
        df.drop(columns=cols_to_drop, inplace=True)
        if "model" in df.columns:
            df.insert(
                loc=1, column="model", value=df.pop("model")
            )  # insert mdl col on 2nd position in df

    if save_dir:
        source_df.to_csv(save_dir / f"{dataset}_source.csv")
        completions_df.to_csv(save_dir / f"{dataset}_completions.csv")

    return source_df, completions_df


def input_parse():
    parser = ArgumentParser()

    # add dataset as arg
    parser.add_argument(
        "-d",
        "--dataset",
        default="dailymail_cnn",
        help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'",
        type=str,
    )

    # flags to only process either human or ai (e.g., if flag -human only is used, then only human will be processed)
    parser.add_argument("-human_only", default=False, action="store_true")
    parser.add_argument("-ai_only", default=False, action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = input_parse()

    # define paths
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets_files" / "text" / "ai_datasets" / "clean_data"
    human_dir = path.parents[2] / "datasets_files" / "text" / "human_datasets"

    metrics_path = path.parents[2] / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    # get cores for multiprocessing (-1 for safety)
    n_cores = mp.cpu_count() - 1
    batch_size = 100

    # HUMAN PROCESSING
    if args.human_only:
        print(f"[INFO:] Processing HUMAN dataset for '{args.dataset}'")
        source_df, completions_df = get_human_metrics(
            human_dir=human_dir,
            dataset=args.dataset,
            batch_size=batch_size,
            n_process=n_cores,
            compute_perplexity=True,  
            save_dir=metrics_path / "human_metrics",
        )

    else:
        # AI PROCESSING
        for temp in [1, 1.5]:
            print(f"[INFO]: Processing AI datasets for '{args.dataset}'")
            ai_savefile = (
                metrics_path
                / "ai_metrics"
                / f"{args.dataset}_completions_temp{temp}.csv"
            )

            ai_metrics_df = get_ai_metrics(
                ai_dir=ai_dir,
                dataset=args.dataset,
                temp=temp,
                batch_size=batch_size,
                compute_perplexity=True,  # for now until we decide on a model
                n_process=n_cores,
                save_dir=metrics_path / "ai_metrics",
            )

        if not args.ai_only:  # if args_ai_only not specified, then run human also!
            print(f"Processing HUMAN datasets for '{args.dataset}'")
            source_df, completions_df = get_human_metrics(
                human_dir=human_dir,
                dataset=args.dataset,
                batch_size=20,
                n_process=n_cores,
                compute_perplexity=True,  # for now until we decide on a model
                save_dir=metrics_path / "human_metrics",
            )


if __name__ == "__main__":
    main()
