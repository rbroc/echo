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

    for split in ["train", "val", "test"]:
        text_df = pd.read_parquet(text_path / f"{split}_text.parquet")
        metrics_df = pd.read_parquet(metrics_path / f"{split}_metrics.parquet")

        print(text_df[["id", "model"]].equals(metrics_df[["id", "model"]]))


if __name__ == "__main__":
    main()
