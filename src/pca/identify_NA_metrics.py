'''
Identifies metrics that contain NA and a threshold of 0 values across the metrics of ALL datasets.

Since PCA cannot handle NA, we need to identify rows with NA metrics. In addition, we are also removing high percentage of 0s since they are not informative.
Even though we are creating seperate classifiers for each dataset, we still want to compare PCA components across datasets. Therefore, the same metrics should be dropped across all datasets.
'''
import pathlib
import pandas as pd

def identify_NA_metrics(df, percent_zero:float=None):
    '''
    Identify rows with NA metrics (and alternatively also high percentage of 0s)

    Args:
        df: dataframe to check
        percent_zero: threshold for percentage of 0s in a column to be considered for removal. Default is None (keep cols with many 0s) 
    '''
    # all na_cols 
    na_cols = df.columns[df.isna().any()].tolist()

    # check for NA values 
    if percent_zero is not None:
        if percent_zero < 0 or percent_zero > 1: # check if percent_zero is either 0, 1 or in between 
            raise ValueError("percent_zero must be either 0 or between 0 and 1")

        zero_cols = [col for col in df.columns if df[col].eq(0).sum() / len(df) >= percent_zero]
    else:
        zero_cols = []

    return na_cols + zero_cols

def main():
    path = pathlib.Path(__file__)
    temp = 1

    datapath = (
        path.parents[2]
        / "datasets_complete"
        / "metrics"
        / f"temp_{temp}"
    )

    # load train metrics data
    train_df = pd.read_parquet(datapath / "train_metrics.parquet")
    val_df = pd.read_parquet(datapath / "val_metrics.parquet")
    test_df = pd.read_parquet(datapath / "test_metrics.parquet")

    # combine
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)


    # drop type cols first (since they are not what should determine what features to drop)
    type_cols = ["model", "id", "is_human", "sample_params", "temperature", "prompt_number", "dataset", "annotations"] 
    df = df.drop(columns=type_cols)

    # filter metrics
    cols = identify_NA_metrics(df, percent_zero=None)

    print(f"[INFO:] Columns to drop: {', '.join(cols)}")


if __name__ == "__main__":
    main()