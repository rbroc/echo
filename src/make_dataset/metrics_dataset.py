"""
Create metrics dataset (to match text dataset)
"""

import pathlib

import pandas as pd
from tqdm import tqdm

from util_split_data import split_on_unique_ids

DATASETS = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]


def load_file(file):
    """
    helper function to load a single file and add dataset column if not present
    """
    df = pd.read_csv(file, index_col=[0])

    if "dataset" not in df.columns:
        if "dailymail_cnn" in file.name:
            df["dataset"] = "dailymail_cnn"
        else:
            df["dataset"] = file.name.split("_")[0]

    return df


def preprocess_metrics(
    ai_dir: pathlib.Path,
    human_dir: pathlib.Path,
    temp: float = 1,
):
    """
    Load metrics

    Args:
        ai_dir: path to directory with ai metrics
        human_dir: path to directoy with human metrics
        temp: temperature of generations.
    """

    all_dfs = []

    for dataset in tqdm(DATASETS, desc="Loading metrics datasets"):
        ai_df = load_file(ai_dir / f"{dataset}_completions_temp{temp}.csv")
        human_df = load_file(human_dir / f"{dataset}_completions.csv")

        dataset_df = pd.concat([human_df, ai_df], ignore_index=True, axis=0)
        all_dfs.append(dataset_df)

    # combine
    if len(all_dfs) > 1:
        final_df = pd.concat(all_dfs, ignore_index=True)
    else:
        final_df = all_dfs[0]

    # add binary outcome for classification (human = 1, ai = 0)
    final_df["is_human"] = final_df["model"].apply(lambda x: 1 if x == "human" else 0)

    # reset index, add unique id col to first col
    final_df = final_df.reset_index(drop=True)
    final_df.insert(0, "unique_id", range(0, len(final_df)))

    return final_df


def main():
    temp = 1

    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets_files" / "metrics" / "ai_metrics"
    human_dir = path.parents[2] / "datasets_files" / "metrics" / "human_metrics"
    outpath = path.parents[2] / "datasets_complete" / "metrics" / f"temp_{temp}"
    outpath.mkdir(parents=True, exist_ok=True)

    df = preprocess_metrics(human_dir=human_dir, ai_dir=ai_dir, temp=temp)

    # sort dataset by id (to ensure consistency with text data, will be shuffled anyways)
    df = df.sort_values(["id", "model"])

    # drop "unique_id" column (if it exists)
    if "unique_id" in df.columns:
        df = df.drop(columns=["unique_id"])

    # split on unique ids
    train_df, val_df, test_df = split_on_unique_ids(df)

    for df in zip([train_df, val_df, test_df], ["train", "val", "test"]):
        df[0].to_parquet(outpath / f"{df[1]}_metrics.parquet")

        # info msgs
        print(f"[INFO]: {df[1]} length: {len(df[0])}")

        print("[INFO:] DATASET COUNTS")
        print(df[0]["dataset"].value_counts())

        print("[INFO:] MODEL COUNTS")
        print(df[0]["model"].value_counts())


if __name__ == "__main__":
    main()
