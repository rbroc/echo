"""
Present txt file as table

Code from: https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/src/modules/visualisation.py
"""

import pathlib
import argparse
import pandas as pd

from great_tables import GT, style, loc, md
from typing import List


def create_df_from_clf_txt(filepath: pathlib.Path, skiprows=[0, 1, 2], nrows=5):
    """
    Create a dataframe from a text file containing the classification report from sklearn.metrics.classification_report

    Args:
        filepath: path to text file
        skiprows: rows to skip when reading the file (e.g. if there is text above the table)
        nrows: number of rows to read from the file

    Returns:
        df: dataframe containing the classification report
    """
    # read only (skip first three rows and last three rows)
    df = pd.read_csv(filepath, skiprows=skiprows, nrows=nrows)

    # replace macro avg and weighted avg with macro_avg and weighted_avg
    df.iloc[:, 0] = df.iloc[:, 0].str.replace(
        r"(macro|weighted)\savg", r"\1_avg", regex=True
    )

    # split the columns by whitespace
    df = df.iloc[:, 0].str.split(expand=True)

    # define new column names
    new_cols = ["class", "precision", "recall", "f1", "support"]
    df.columns = new_cols

    # identify the row with the accuracy score
    is_accuracy = df["class"] == "accuracy"

    # move the accuracy row values into the precision and recall columns (they are placed incorrectly when the columns are split)
    df.loc[is_accuracy, ["f1", "support"]] = df.loc[
        is_accuracy, ["precision", "recall"]
    ].values

    # set precision and recall to None for the accuracy row
    df.loc[is_accuracy, ["precision", "recall"]] = None

    # transpose
    df = df.T

    # name columns (where 0 and 1 refers to the classes)
    df.columns = ["0", "1", "accuracy", "macro_avg", "weighted_avg"]

    # drop class index
    df = df[~df.index.isin(["class"])]

    return df


def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg
    parser.add_argument(
        "-d",
        "--dataset",
        default="dailymail_cnn",
        help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'",
        type=str,
    )
    parser.add_argument(
        "-t", "--temp", default=1, help="Temperature of generations", type=float
    )

    args = parser.parse_args()

    return args


def prepare_df_for_table(df, type: str):
    # avg_df
    avg_df = df[["accuracy", "macro_avg", "weighted_avg"]]
    print(avg_df)

    # class df
    class_df = df[["0", "1"]]

    # unstack
    class_df = class_df.unstack().to_frame().T

    # rename so that columns are called 0_precision, 0_recall, 0_f1-score, 0_support, 1_precision, etc.
    class_df.columns = [f"{col}_{row}" for row, col in class_df.columns]

    # add type col
    class_df["type"] = type

    # make empty col that has None for all values )
    class_df["empty_1"] = None  # after human
    class_df["empty_0"] = None  # after synthetic

    # sort cols
    class_df = class_df[
        [
            "type",
            # human
            "f1_1",
            "precision_1",
            "recall_1",
            "support_1",
            "empty_1",
            # synthetic
            "f1_0",
            "precision_0",
            "recall_0",
            "support_0",
            "empty_0",
        ]
    ]

    # add overall scores
    class_df["accuracy_f1"] = avg_df.loc["f1", "accuracy"]
    class_df["macro_avg_f1"] = avg_df.loc["f1", "macro_avg"]

    # another jank way to make empty space after the overall scores
    class_df["empty_2"] = None

    # rename f1-score to f1_score
    class_df.columns = class_df.columns.str.replace("-", "_")

    return class_df


def create_rowgroups(df: pd.DataFrame):
    # set default value for group
    df["group"] = "Other - Human"

    # define a dictionary for type-to-group mapping
    type_to_group = {
        "all_models": "All Models - Human",
        "beluga7b": "Beluga 7b - Human",
        "llama2_chat13b": "Llama2 Chat 13b - Human",
        "llama2_chat7b": "Llama2 Chat 7b - Human",
        "mistral": "Mistral 7b - Human",
    }

    # apply group assignment based on partial matches
    for key, group in type_to_group.items():
        df.loc[df["type"].str.contains(key, case=False, na=False), "group"] = group

    # mv group to first column
    df = df[["group"] + [col for col in df.columns if col != "group"]]

    # new row names
    row_names = {
        "all_models_all_features": "All Features",
        "all_models_embeddings": "Embeddings",
        "all_models_tfidf_1000_features": "TF-IDF 1000 Features",
        "all_models_top3_features": "Top 3 Features",
        "beluga7b-human_all_features": "All Features",
        "llama2_chat13b-human_all_features": "All Features",
        "llama2_chat7b-human_all_features": "All Features",
        "mistral7b-human_all_features": "All Features",
    }

    # apply new row names with map
    df["type"] = df["type"].map(row_names)

    return df


def create_table(df, title: str, savepath: pathlib.Path, subtitle:str=None):
    """
    create table from df using great_tables

    Args:
        df: dataframe
        title: title of the table
        subtitle: subtitle of table
        savepath: path to save the table
    """
    # create row groups
    df = create_rowgroups(df)

    table = GT(df, rowname_col="type", groupname_col="group")

    # make multi-col spanners
    table = table.tab_spanner(
        label=md("**Human**"), columns=["f1_1", "precision_1", "recall_1", "support_1"]
    ).tab_spanner(
        label=md("**Synthetic**"), columns=["f1_0", "precision_0", "recall_0", "support_0"]
    ).tab_spanner(
        label=md("**Overall**"), columns=["accuracy_f1", "macro_avg_f1"]
    )

    # rename cols to only represent the class
    table = table.cols_label(
        precision_0="Precision",
        precision_1="Precision",
        recall_0="Recall",
        recall_1="Recall",
        f1_0="F1",
        f1_1="F1",
        support_0="Support",
        support_1="Support",
        accuracy_f1="Accuracy (F1)",
        macro_avg_f1="Macro Avg (F1)",
        type="Type",
        # hacky way to add space between two classes and after overall
        empty_1="",
        empty_0="",
        empty_2=""
    )

    # layout
    table = (
        table.opt_vertical_padding(scale=1.5).opt_horizontal_padding(scale=2)
        # center all but the type column
        .cols_align("center")
    )

    # annotations
    table = table.tab_header(title, subtitle)

    # style
    table = table.tab_style(
        style=[style.text(weight="bold"), style.fill(color="#E8E8E8")],
        locations=loc.row_groups(),
    )

    # save table
    html = table.as_raw_html("table.html")

    # save as html
    with open(savepath / "all_results.html", "w") as f:
        f.write(html)


def main():
    args = input_parse()

    path = pathlib.Path(__file__)

    dataset, temp = args.dataset, args.temp

    if temp == 1.0:
        temp = int(temp)

    savepath = path.parents[3] / "results" / "classify" / "clf_results"
    datapath = savepath / "clf_reports" / f"{dataset}_temp{temp}"

    dfs = []

    # get filepaths
    filepaths = [file for file in datapath.iterdir() if file.suffix == ".txt"]

    # sort filepaths by name
    filepaths = sorted(filepaths, key=lambda x: x.stem)

    for file in filepaths:
        df = create_df_from_clf_txt(file)
        table_df = prepare_df_for_table(df, file.stem)
        dfs.append(table_df)

    final_df = pd.concat(dfs)

    # create table
    create_table(
        df=final_df,
        title=f"{dataset.capitalize()}",
        subtitle = "(Validation Data)",
        savepath=datapath,
    )


if __name__ == "__main__":
    main()
