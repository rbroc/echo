'''
Identifies metrics with 90% NA or 0 values across the metrics of ALL datasets.

Since PCA cannot handle NA, we need to identify rows with NA metrics. In addition, we are also removing high percentage of 0s since they are not informative.
Even though we are creating seperate classifiers for each dataset, we still want to compare PCA components across datasets. Therefore, the same metrics should be dropped across all datasets.
'''
import pathlib
import sys
import pandas as pd
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.process_metrics import load_metrics

def identify_NA_metrics(df, percent_NA=0.9, percent_zero=0.8):
    '''
    Identify rows with NA metrics or high percentage of 0s
    '''
    # check if percent_NA and percent_zero are either 0, 1 or in between
    if percent_NA < 0 or percent_NA > 1 or percent_zero < 0 or percent_zero > 1:
        raise ValueError("percent_NA and percent_zero must be either 0 or between 0 and 1")

    identified_cols = []

    # check for NA values 
    for col in df.columns:
        if df[col].isna().sum() / len(df) >= percent_NA or df[col].eq(0).sum() / len(df) >= percent_zero:
            identified_cols.append(col)

    return identified_cols

def main():
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"

    df = load_metrics(
                            human_dir=datapath / "human_metrics", 
                            ai_dir=datapath / "ai_metrics",
                            dataset="stories", temp=1.5, 
                            human_completions_only=True,
                            filter_lengths=True
                            )

    # filter metrics
    cols = identify_NA_metrics(df, percent_NA=0.9, percent_zero=0.9)

    print(cols)


if __name__ == "__main__":
    main()