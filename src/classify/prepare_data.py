'''
Prepare data for classifiction
'''
import pathlib
import pandas as pd

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.generate.generation import extract_min_max_tokens 

def load_file(file):
    '''helper function to load a single file and add dataset column if not present'''
    df = pd.read_csv(file, index_col=[0])
    
    if "dataset" not in df.columns:
        if "dailymail_cnn" in file.name:
            df["dataset"] = "dailymail_cnn"
        else: 
            df["dataset"] = file.name.split("_")[0]
    
    return df

def load_human_metrics(human_dir: pathlib.Path, dataset:str = None, human_completions_only=True):
    '''
    load human metrics

    Args:
        human_dir: path to directory with human metrics
        dataset: name of dataset to load. If None, loads all datasets.
        human_completions_only: whether to load only human completions. If False, loads also files ending with "source.csv" which contains metrics for the source text (prompt col) instead of completions.

    Returns:
        dfs: list of dataframes
    '''
    # get file paths for human metrics, if dataset is specified, filter by dataset
    file_paths = [file for file in human_dir.iterdir() if dataset is None or dataset in file.name]

    if human_completions_only:
        print("[WARNING]: Loading only human completions... If you want to load 'source' metrics also, set human_completions_only=False.")
        
        # filter to get only completions
        file_paths = [file for file in file_paths if "completions" in file.name]

    else: 
        # get all file paths 
        file_paths = [file for file in file_paths]

    # sort file paths
    file_paths = sorted(file_paths)

    # load all files into a list of dfs
    dfs = [load_file(file) for file in file_paths]

    return dfs

def load_ai_metrics(ai_dir: pathlib.Path, dataset:str = None, temp:float = None):
    '''
    load ai metrics

    Args:
        ai_dir: path to directory with ai metrics
        dataset: name of dataset to load. If None, loads all datasets.
        temp: temperature of generations. If None, loads all temperatures.

    Returns:
        dfs: list of dataframes
    '''
    # get file paths for ai metrics, if dataset is specified, filter by dataset and temperature if specified
    file_paths = [file for file in ai_dir.iterdir() if dataset is None or dataset in file.name]

    # if temperature is specified, filter by temperature
    if temp:
        file_identifier = f"{temp}.csv"
        print(f"[INFO:] Loading only AI data for temperature {temp} ...")
        file_paths = [file for file in file_paths if file.name.endswith(file_identifier)]

        if len(file_paths) == 0:
            raise ValueError(f"No files found for temperature {temp}.")
    
    # sort file paths
    file_paths = sorted(file_paths)

    # load all files into a list of dfs
    dfs = [load_file(file) for file in file_paths]
  
    return dfs


def load_metrics(human_dir: pathlib.Path, ai_dir:pathlib.Path, human_completions_only=True, dataset: str = None, temp:float= None):
    '''
    Load metrics

    Args:
        data_dir: path to directory with metrics
        dataset: name of dataset to load. If None, loads all datasets.
        temp: temperature of generations. If None, loads all temperatures.
    '''
    # load multiple files 
    human_dfs = load_human_metrics(human_dir, dataset=dataset, human_completions_only=human_completions_only)
    ai_dfs = load_ai_metrics(ai_dir, dataset=dataset, temp=temp)

    all_dfs = human_dfs + ai_dfs

    # combine all loaded dfs into a single df
    final_df = pd.concat(all_dfs, ignore_index=True)

    # add binary outcome column for classification (human = 1, ai = 0)
    final_df["is_human"] = final_df["model"].apply(lambda x: 1 if x == "human" else 0)

    # reset index, add unique id col to first col 
    final_df = final_df.reset_index(drop=True)
    final_df.insert(0, "unique_id", range(0, len(final_df)))

    return final_df

def drop_lengths(df, verbose=True):
    '''
    Drop rows based on min and max doc lengths
    '''
    # extract min and max tokens from completions, drop cols that are below or above these in doc length
    min_tokens, max_tokens = extract_min_max_tokens(df["dataset"].unique()[0])

    # drop cols that are below or above min and max tokens
    filtered_df = df[(df["doc_length"] >= min_tokens) & (df["doc_length"] <= max_tokens)]

    # print info msg
    if verbose:
        print(f"[INFO:] Dropped {len(df) - len(filtered_df)} rows based on doc length. Min tokens: {min_tokens}, Max tokens: {max_tokens}")
        print(f"[INFO:] Min len in df: {filtered_df['doc_length'].min()}, Max len in df: {filtered_df['doc_length'].max()}")

    return filtered_df

def drop_metrics(df, percent_NA=0.8, percent_zero=0.8, verbose=True, log_file:pathlib.Path=None):
    '''
    Drop metric columns where there are more than the percent threshold observations that are NA (percent_NA) and less than the percent threshold observations that are zero (percent_zero)

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
    cols_to_keep = ["id", "unique_id", "sample_params", "temperature", "prompt_number", "is_human"] 

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

def filter_metrics(df, percent_NA=0.8, percent_zero=0.8, verbose=True, log_file:pathlib.Path=None):
    '''
    Filter data based on NA and zero values in the columns and doc length
    '''
    # remove rows with NA values in doc length
    filtered_lengths = drop_lengths(df, verbose=verbose)

    # remove cols with high NA and zero values
    filtered_metrics = drop_metrics(filtered_lengths, percent_NA=percent_NA, percent_zero=percent_zero, verbose=verbose, log_file=log_file)

    return filtered_metrics

def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"

    df = load_metrics(
                            human_dir=datapath / "human_metrics", 
                            ai_dir=datapath / "ai_metrics",
                            dataset="stories", temp=1.5, 
                            human_completions_only=True
                            )

    # filter metrics
    df = filter_metrics(df, percent_NA=0.9, percent_zero=0.9, verbose=True)


if __name__ == "__main__":
    main()