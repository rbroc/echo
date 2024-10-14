"""
create correlation heat maps 
"""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.cols_to_drop import get_cols_to_drop


def create_corrM(df, save_dir=None, file_name=None, fontsize=12, fontweight=None):
    # create correlation matrix
    corrM = df.corr()

    # plot
    sns.set_theme(rc={"figure.figsize": (20, 20)})
    # drop na from corrM
    corrMplot = sns.heatmap(corrM.dropna(how="all"))

    # make labels bigger and bold
    plt.xticks(fontsize=fontsize, fontweight=fontweight)
    plt.yticks(fontsize=fontsize, fontweight=fontweight)

    # tight
    plt.tight_layout()

    # save plot
    if save_dir and file_name:
        save_dir.mkdir(parents=True, exist_ok=True)
        corrMplot.get_figure().savefig(save_dir / file_name)

    # close plot
    plt.close()

    return corrM, corrMplot


def main():
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "datasets_complete" / "metrics"

    temp = 1

    # dirs based on temp
    savedir = path.parents[2] / "results" / "heatmaps_metrics"
    savedir.mkdir(parents=True, exist_ok=True)

    datapath = (
        path.parents[2]
        / "datasets_complete"
        / "metrics"
        / f"temp_{temp}"
        / "train_metrics.parquet"
    )

    # filter to only include one dataset
    train_df = pd.read_parquet(datapath)

    ## DEFINE FEATURES ##
    all_features = train_df.columns.tolist()
    cols_to_drop = get_cols_to_drop()
    features = [feat for feat in all_features if feat not in cols_to_drop]

    create_corrM(
        train_df[features],
        savedir,
        f"heatmap_temp{temp}.png",
        fontsize=12,
        fontweight=None,
    )

    # subset features
    subset_features = [
        "alpha_ratio",
        "proportion_unique_tokens",
        "n_sentences",
        "n_unique_tokens",
        "doc_length",
        "rix",
        "gunning_fog",
        "pos_prop_NOUN",
        "pos_prop_NUM",
        "perplexity",
        "sentence_length_mean",
        "proportion_ellipsis",
        "pos_prop_SCONJ",
        "syllables_per_token_mean",
        "dependency_distance_mean",
        "pos_prop_VERB",
        "flesch_reading_ease",
        "mean_word_length",
        "oov_ratio",
        "entropy",
    ]
    subset_features = sorted(subset_features)

    create_corrM(
        train_df[subset_features],
        savedir,
        f"heatmap_temp{temp}_subset.png",
        fontsize=18,
        fontweight="bold",
    )


if __name__ == "__main__":
    main()
