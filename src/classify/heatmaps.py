'''
create correlation heat maps 
'''
import argparse
import pathlib
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from prepare_data import load_metrics

def filter_metrics(df, percent_NA=0.8, percent_zero=0.8, verbose=True, log_file:pathlib.Path=None):
    '''
    Remove metric columns where there are more than the percent threshold observations that are NA (percent_NA) and less than the percent threshold observations that are zero (percent_zero)

    Args: 
        df: dataframe with metrics to filter
        percent_NA: threshold for NA values
        percent_zero: threshold for zero values
        verbose: print out which columns have been dropped
        log_file: path to logfile which will log which columns have been dropped and the remaining columns. Default is None, which means no logging.

    Returns:
        filtered_df: dataframe with filtered columns
    '''
    # identify cols that must not be dropped regardless of NA values (as they include ID cols or with cols that are always NA if model = human and would thus be dropped if we filter by a HIGH NA threshold)
    cols_to_keep = ["id", "unique_id", "sample_params", "temperature", "prompt_number"] 

    # identify cols to filter based on the cols to keep
    cols_to_filter = [col for col in df.columns if col not in cols_to_keep]

    # check if percent_NA and percent_zero are either 0, 1 or in between
    if percent_NA < 0 or percent_NA > 1 or percent_zero < 0 or percent_zero > 1:
        raise ValueError("percent_NA and percent_zero must be either 0 or between 0 and 1")
    
    # get number of observations for setting NA threshold
    n_obs = len(df)

    # drop COLUMNS where more than percent threshold observations are NA (percent_NA)
    if percent_NA > 0:
        filtered_df = df[cols_to_filter].dropna(axis=1, thresh=n_obs * percent_NA) 
    else:
        filtered_df = df[cols_to_filter]
        print("[INFO:] No columns dropped based on NA threshold")
    
    # drop COLUMNS where less than percent threshold observations are zero (percent_zero), see https://stackoverflow.com/questions/44250642/drop-columns-with-more-than-70-zeros
    if percent_zero > 0:
        filtered_df = filtered_df.loc[:, (filtered_df == 0).mean() < percent_zero]
    else: 
        print("[INFO:] No columns dropped based on zero threshold")

    # join with df with cols to keep
    filtered_df = df[cols_to_keep].join(filtered_df)

    # identify which cols have been dropped
    dropped_cols = [col for col in df.columns if col not in filtered_df.columns]
    if verbose and len(filtered_df.columns) != len(df.columns):
        print(f"[INFO:] Columns dropped: {dropped_cols}")

    if log_file and len(filtered_df.columns) != len(df.columns):
        dataset = df["dataset"].unique()[0]
        temp = df["temperature"].unique()[1] # as the first is always nan (for human)

        with open(log_file, "a") as f:
            f.write(f"Dataset: {dataset}, Temp: {temp}\n")
            f.write(f"Columns dropped: {dropped_cols}\n")
            f.write(f"Dropping criteria: {int(percent_NA*100)}% of values in col were either NA or {int(percent_zero*100)}% of values were zero\n")
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
            df = filter_metrics(df, percent_NA=0.9, percent_zero=0.9, verbose=False, log_file=path.parents[0] / "prelim_plots" / "filtered_metrics_log.txt")

            create_corrM(df, f"{dataset.upper()} (Temperature of {temp})" , path.parents[0] / "prelim_plots", f"{dataset}_heatmap_temp{temp}.png")


if __name__ == "__main__":
    main()