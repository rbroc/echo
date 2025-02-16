"""
Make text dataset: 
- load individual files for ai and human data 
- combine ai and human data (already cleaned)
- split into train, val, test
- save splits

Run this script
    python src/make_dataset/text_dataset.py -t {TEMPERATURE}

The {TEMPERATURE} could be either 1 or 1.5. Defaults to 1.
"""
import argparse

import pathlib

import pandas as pd
from tqdm import tqdm
from util_split_data import split_on_unique_ids

MODELS = ["beluga7b", "llama2_chat7b", "llama2_chat13b", "mistral7b"]
PROMPT_NUMBERS = [21]
DATASETS = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

def input_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--temp", default=1, help="Temperature of generations", type=float
    )

    args = parser.parse_args()
    return args

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

    ai_paths = sorted(ai_paths)

    return ai_paths


def get_all_paths(
    ai_dir: pathlib.Path,
    human_dir: pathlib.Path,
    dataset: str = "dailydialog",
    temp: float | int = 1,
) -> list[pathlib.Path]:
    """
    Get all paths pertaining to a particular dataset
    """
    valid_datasets = [d.name for d in human_dir.iterdir()]
    if dataset not in valid_datasets:
        raise ValueError(f"Dataset {dataset} not found in {human_dir}")

    ai_paths = get_ai_paths(ai_dir=ai_dir, dataset=dataset, temp=temp)
    human_path = human_dir / dataset / "data.ndjson"

    return ai_paths, human_path


def load_dataset(
    ai_paths: list[pathlib.Path], human_path: pathlib.Path
) -> tuple[pd.DataFrame]:
    """
    Load data from paths extracted from get_paths function
    """
    ai_dfs = [pd.read_json(p, lines=True) for p in ai_paths]
    human_df = pd.read_json(human_path, lines=True)

    return ai_dfs, human_df


def standardize_human_data(human_df):
    # add model and is_human cols
    human_df["model"] = "human"
    human_df["is_human"] = 1

    # drop source col to make smaller file
    human_df.drop(columns=["source"], inplace=True)
    human_df.rename(columns={"human_completions": "completions"}, inplace=True)

    return human_df


def preprocess_datasets(
    ai_dir: pathlib.Path,
    human_dir: pathlib.Path,
    temp: int | float = None,
):
    """
    Loads and prepares as many datasets as needed

    Args:
        ai_dir: path to directory with AI datasets
        human_dir: path to directory with human datasets
        temp: temperature in file name (e.g., 1 if temp1, 1.4 if temp1.4)

    Returns:
        all_dfs_combined: combined dataframe with all datasets
    """
    all_dfs = []

    for dataset in tqdm(DATASETS, desc="Datasets"):
        ai_paths, human_path = get_all_paths(ai_dir, human_dir, dataset, temp)
        ai_dfs, human_df = load_dataset(ai_paths, human_path)

        human_df = standardize_human_data(human_df)

        # combine
        dataset_df = pd.concat([human_df, *ai_dfs], ignore_index=True, axis=0)
        dataset_df["dataset"] = dataset

        all_dfs.append(dataset_df)

    if len(all_dfs) > 1:
        final_df = pd.concat(all_dfs, ignore_index=True, axis=0)

    else:
        final_df = all_dfs[0]

    return final_df


def main():
    args = input_parse()
    temp = args.temp

    if (
        temp == 1.0
    ):  # when temp is specified from CLI, it is a float (1.0), so needs to be converted (while allowing for 1.5 as input)
        temp = int(temp)

    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets_files" / "text" / "ai_datasets" / "clean_data"
    human_dir = path.parents[2] / "datasets_files" / "text" / "human_datasets"
    outpath = path.parents[2] / "datasets_complete" / "text" / f"temp_{temp}"
    outpath.mkdir(parents=True, exist_ok=True)

    df = preprocess_datasets(
        ai_dir,
        human_dir,
        temp=temp,
    )

    # sort dataset by id and model (to ensure consistency with metrics data, will be shuffled anyways)
    df = df.sort_values(["id", "model"])

    # split on unique ids
    train_df, val_df, test_df = split_on_unique_ids(df)

    for df in zip([train_df, val_df, test_df], ["train", "val", "test"]):
        df[0].to_parquet(outpath / f"{df[1]}_text.parquet")

        # info msgs
        print(f"[INFO]: {df[1]} length: {len(df[0])}")

        print("[INFO:] DATASET COUNTS")
        print(df[0]["dataset"].value_counts())

        print("[INFO:] MODEL COUNTS")
        print(df[0]["model"].value_counts())


if __name__ == "__main__":
    main()
