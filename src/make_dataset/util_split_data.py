"""
Split data into train, test, and validation sets
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split


def split_on_unique_ids(
    df: pd.DataFrame, random_state: int = 129, val_test_size: float = 0.15
):
    """
    split data into train, test, and validation sets based on unique ids
    """
    # get a DF only containing the unique ids per dataset
    unique_ids_df = df[["id", "dataset"]].drop_duplicates(subset=["id", "dataset"])

    # split these unique ids, stratify by dataset to keep even
    train_ids, test_val_ids = train_test_split(
        unique_ids_df,
        test_size=0.15 * 2,
        random_state=random_state,
        stratify=unique_ids_df["dataset"],
    )
    val_ids, test_ids = train_test_split(
        test_val_ids,
        test_size=0.5,
        random_state=random_state,
        stratify=test_val_ids["dataset"],
    )

    # extract data
    train_df = df[df["id"].isin(train_ids["id"])]
    val_df = df[df["id"].isin(val_ids["id"])]
    test_df = df[df["id"].isin(test_ids["id"])]

    # shuffle data (otherwise all parallel texts of same ID (e.g., 'dailydioalog-6202') will be right after each other) - sampling w. frac=1 to keep all
    train_df = train_df.sample(frac=1, random_state=random_state)
    val_df = val_df.sample(frac=1, random_state=random_state)
    test_df = test_df.sample(frac=1, random_state=random_state)

    return train_df, val_df, test_df
