"""
Make dataset: 
- preprocess ai data, combine with human data
- split into train, val, test
- save splits
"""

import ast
import pathlib
import re
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.split_data import create_split

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


def clean_ai_df(df, col="completions"):
    """
    lowercase, remove irregular format to standardise to human datasets like in src/clean/clean_data.py
    """
    # remove space at the beginning of a string
    df[col] = df[col].str.replace(r"^\s+", "", regex=True)

    # make lowercase ALL
    df[col] = df[col].str.lower()

    # convert any a: or b: in the start to A: and B: (to match human data)
    df[col] = df[col].str.replace(r"^(a:|b:)", lambda m: m.group(1).upper(), regex=True)

    # rm newline characters
    df[col] = df[col].str.replace("<newline>", " ", regex=False)

    # rm extra spaces
    df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    return df


def standardize_ai_data(ai_dfs: list[pd.DataFrame], clean: bool = True):
    """
    Standardize AI data to match human data.

    Args:
        ai_dfs: list of dataframes
        clean_ai: whether to clean ai data (e.g., lowercase, removing irregular format)

    Returns:
        ai_dfs: list of dataframes
    """
    for idx, df in enumerate(ai_dfs):
        new_df = df.copy()

        new_df["is_human"] = 0

        # standardise prompt and completions cols
        prompt_colname = [col for col in new_df.columns if col.startswith("prompt_")][
            0
        ]  # get column name that starts with prompt_ (e.g., prompt_1, prompt_2, ...)
        new_df["prompt_number"] = prompt_colname.split("_")[1]  # extract numbers
        new_df.rename(columns={prompt_colname: "prompt"}, inplace=True)

        mdl_colname = [col for col in new_df.columns if col.endswith("_completions")][0]
        new_df["model"] = re.sub(
            r"_completions$", "", mdl_colname
        )  # remove "_completions" from e.g., "beluga_completions"
        new_df.rename(columns={mdl_colname: "completions"}, inplace=True)

        # add temperature val to col
        if "sample_params" in df.columns:
            new_df["temperature"] = df.sample_params.apply(
                lambda x: (
                    ast.literal_eval(x).get("temperature") if pd.notna(x) else None
                )
            )

        if clean:
            new_df = clean_ai_df(new_df)

        ai_dfs[idx] = new_df  # replace original df with new df

    return ai_dfs


def standardize_human_data(human_df):
    # add model and is_human cols
    human_df["model"] = "human"
    human_df["is_human"] = 1

    # drop source col to make smaller file
    human_df.drop(columns=["source"], inplace=True)
    human_df.rename(columns={"human_completions": "completions"}, inplace=True)

    return human_df


def combine_data(ai_dfs, human_df):
    """
    Return a dataframe for a particular dataset with all AI generations and human data in one.

    Args:
        ai_dfs: list of dataframes
        human_df: dataframe corresponding to the dfs in ai_dfs

    Returns:
        combined_df: combined dataframe
    """
    ai_dfs = standardize_ai_data(ai_dfs, clean=True)
    human_df = standardize_human_data(human_df)

    # combine
    all_dfs = [human_df, *ai_dfs]
    combined_df = pd.concat(all_dfs, ignore_index=True, axis=0)

    return combined_df


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
        models: list of models to include
        datasets: list of datasets to include
        clean_ai: whether to clean ai data (e.g., lowercase, removing irregular format)

        temp: temperature in file name (e.g., 1 if temp1, 1.4 if temp1.4)
        prompt_n: prompt number (e.g., 3 or 21). Defaults to 21 which is what we settled on.

    Returns:
        all_dfs_combined: combined dataframe with all datasets
    """
    all_dfs = []

    for dataset in tqdm(DATASETS, desc="Datasets"):
        ai_paths, human_path = get_all_paths(ai_dir, human_dir, dataset, temp)
        ai_dfs, human_df = load_dataset(ai_paths, human_path)
        dataset_df = combine_data(ai_dfs, human_df)
        dataset_df["dataset"] = dataset

        all_dfs.append(dataset_df)

    if len(all_dfs) > 1:
        final_df = pd.concat(all_dfs, ignore_index=True, axis=0)

    else:
        final_df = all_dfs[0]

    return final_df


def main():
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"
    human_dir = path.parents[2] / "datasets" / "human_datasets"

    combined_df = preprocess_datasets(
        ai_dir,
        human_dir,
        temp=1,
    )

    print(combined_df)

    # save as jsonl
    outpath = path.parents[2] / "datasets" / "complete_datasets"
    outpath.mkdir(parents=True, exist_ok=True)

    # split (stratified)
    stratify_cols = ["is_human", "dataset", "model"]
    train_df, val_df, test_df = create_split(
        combined_df, random_state=129, val_test_size=0.15, stratify_cols=stratify_cols
    )

    # save splits as parquet
    for df in zip([train_df, val_df, test_df], ["train", "val", "test"]):
        df[0].to_parquet(outpath / f"{df[1]}_text.parquet")

        # print len
        print(f"[INFO]: {df[1]} length: {len(df[0])}")

        # print length of each dataset (df["dataset"])
        print(df[0]["dataset"].value_counts())


if __name__ == "__main__":
    main()
