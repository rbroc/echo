'''
create correlation heat maps 
'''
import argparse
import pathlib
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from prepare_data import load_metrics, filter_metrics


def create_corrM(df, plot_title, save_dir, file_name):
    # define out irrelvant cols for correlation matrix
    cols_to_drop = ["id", "unique_id", "sample_params", "temperature", "prompt_number"]

    # create correlation matrix
    corrM = df.drop(cols_to_drop, axis=1).corr()

    # plot
    sns.set_theme(rc={'figure.figsize':(20, 20)})
    # drop na from corrM 
    corrMplot = sns.heatmap(corrM.dropna(how="all"))
    corrMplot.set_title(plot_title)

    # save plot
    corrMplot.get_figure().savefig(save_dir / file_name)

    # close plot
    plt.close()

    return corrM, corrMplot


def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"

    for dataset in ["stories", "dailymail_cnn", "mrpc", "dailydialog"]:
        for temp in [1, 1.5]:
            df = load_metrics(
                                    human_dir=datapath / "human_metrics", 
                                    ai_dir=datapath / "ai_metrics",
                                    dataset=dataset, temp=temp, 
                                    human_completions_only=True
            )

            # filter 
            df = filter_metrics(df, percent_NA=0.9, percent_zero=0.9, verbose=False, log_file=path.parents[0] / "prelim_plots" / "heatmaps" / "filtered_metrics_log.txt")

            create_corrM(df, f"{dataset.upper()} (Temperature of {temp})", path.parents[0] / "prelim_plots" / "heatmaps", f"{dataset}_heatmap_temp{temp}.png")


if __name__ == "__main__":
    main()