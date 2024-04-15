'''
create correlation heat maps 
'''
import argparse
import pathlib
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from prepare_data import load_metrics


def filter_metrics(df, percent_NA=0.8, percent_zero=0.8, verbose=True, log_file=None):
    '''
    rm cols where there are less than the percent_NA observations (non-na)
    '''
    # get number of observations
    n_obs = len(df)

    # identify cols that must not be dropped 
    cols_to_keep = ["id", "unique_id", "sample_params", "temperature", "prompt_number"]

    # get subset of cols to use in filtering
    cols_to_filter = [col for col in df.columns if col not in cols_to_keep]
    
    # drop COLUMNS where there are more than the percent threshold observations that are na (percent_NA)
    filtered_df = df[cols_to_filter].dropna(axis=1, thresh=n_obs * percent_NA)

    # drop COLUMNS where there are less than the percent threshold observations that are zero (percent_zero)
    filtered_df = filtered_df.loc[:, (filtered_df == 0).mean() < percent_zero]


    # join with df with cols to keep
    filtered_df = df[cols_to_keep].join(filtered_df)

    # identify which cols have been dropped
    if verbose and len(filtered_df.columns) != len(df.columns):
        dropped_cols = [col for col in df.columns if col not in filtered_df.columns]
        print(f"[INFO:] Columns dropped: {dropped_cols}")

    if log_file:
        dataset = df["dataset"].unique()[0]
        temp = df["temperature"].unique()[1] # as the first is always nan (for human)

        with open(log_file, "a") as f:
            f.write(f"Dataset: {dataset}, Temp: {temp}\n")
            f.write(f"Columns dropped: {dropped_cols}\n")
            f.write(f"Dropping criteria: {percent_NA} NA, {percent_zero} zero\n")
            f.write(f"Remaining columns: {filtered_df.columns}\n\n")
            f.write(f"{'-'*50}\n\n")

    return filtered_df


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
            df = filter_metrics(df, percent_NA=0.9, percent_zero=0.9, verbose=True, log_file=path.parents[0] / "prelim_plots" / "filtered_metrics_log.txt")

            create_corrM(df, f"{dataset.upper()} (Temperature of {temp})" , path.parents[0] / "prelim_plots", f"{dataset}_heatmap_temp{temp}.png")


if __name__ == "__main__":
    main()