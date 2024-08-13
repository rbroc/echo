'''
Functions to load metrics data + drop lengths below or above min and max tokens. 
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
        print(f"[INFO:] {df['dataset'].unique()[0].upper()}: Dropped {len(df) - len(filtered_df)} rows on doc length. Tokens min/max {min_tokens}/{max_tokens}. Final doc length min/max: {int(min(filtered_df['doc_length']))}/{int(max(filtered_df['doc_length']))}")

    return filtered_df

def load_human_metrics(human_dir: pathlib.Path, dataset:str = None, human_completions_only=True, filter_lengths=True):
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

    # filter lengths
    if filter_lengths:
        dfs = [drop_lengths(df, verbose=True) for df in dfs]

    return dfs

def load_ai_metrics(ai_dir: pathlib.Path, dataset:str = None, temp:float = None, filter_lengths=True):
    '''
    load ai metrics

    Args:
        ai_dir: path to directory with ai metrics
        dataset: name of dataset to load. If None, loads all datasets.
        temp: temperature of generations. If None, loads all temperatures.

    Returns:
        dfs: list of dataframes
    '''
    # get file paths for ai metrics, if dataset is specified, filter by datasef
    file_paths = [file for file in ai_dir.iterdir() if dataset is None or dataset in file.name]

    # if temperature is specified, filter by temperature
    if temp:
        file_identifier = f"{temp}.csv"
        print(f"[INFO:] Loading only AI data for temperature {temp} ...")
        file_paths = [file for file in file_paths if file.name.endswith(file_identifier)]

        if len(file_paths) == 0:
            raise ValueError(f"No files found for temperature {temp}.")
    
    file_paths = sorted(file_paths)

    # load all files into a list of dfs
    dfs = [load_file(file) for file in file_paths]

    # filter lengths
    if filter_lengths:
        dfs = [drop_lengths(df, verbose=True) for df in dfs]
  
    return dfs

def load_metrics(human_dir: pathlib.Path, ai_dir:pathlib.Path, filter_lengths=True, human_completions_only=True, dataset: str = None, temp:float= None):
    '''
    Load metrics

    Args:
        data_dir: path to directory with metrics
        dataset: name of dataset to load. If None, loads all datasets.
        temp: temperature of generations. If None, loads all temperatures.
    '''
    # load multiple files 
    human_dfs = load_human_metrics(human_dir, dataset=dataset, human_completions_only=human_completions_only, filter_lengths=filter_lengths)
    ai_dfs = load_ai_metrics(ai_dir, dataset=dataset, temp=temp, filter_lengths=filter_lengths)

    all_dfs = human_dfs + ai_dfs

    # combine all loaded dfs into a single df
    final_df = pd.concat(all_dfs, ignore_index=True)

    # add binary outcome column for classification (human = 1, ai = 0)
    final_df["is_human"] = final_df["model"].apply(lambda x: 1 if x == "human" else 0)

    # reset index, add unique id col to first col 
    final_df = final_df.reset_index(drop=True)
    final_df.insert(0, "unique_id", range(0, len(final_df)))

    return final_df
