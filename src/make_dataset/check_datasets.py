"""
Check whether metrics and text dataset matches
"""

import pathlib

import pandas as pd


def main():
    path = pathlib.Path(__file__)
    temp = 1

    metrics_path = path.parents[2] / "datasets_complete" / "metrics" / f"temp_{temp}"
    text_path = path.parents[2] / "datasets_complete" / "text" / f"temp_{temp}"

    splits = ["train", "val", "test"]

    for split in splits:
        print(f"[INFO]: Checking {split} split")
        text_df = pd.read_parquet(text_path / f"{split}_text.parquet").reset_index(
            drop=True
        )
        metrics_df = pd.read_parquet(
            metrics_path / f"{split}_metrics.parquet"
        ).reset_index(drop=True)

        # print whether they match on id and model
        print(
            f"[INFO]: Ids and model match: {text_df[['id', 'model']].equals(metrics_df[['id', 'model']])}"
        )

        print(f"[INFO]: Length of text df: {len(text_df)}")
        print(f"[INFO]: Length of metrics df: {len(metrics_df)}\n")


if __name__ == "__main__":
    main()
