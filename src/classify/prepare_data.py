'''
Prepare data for classifiction
'''
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

def load_file(file):
    '''helper function to load a single file and add dataset column if not present'''
    df = pd.read_csv(file, index_col=[0])
    
    if "dataset" not in df.columns:
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

def create_split(df, random_state=129, val_test_size:float=0.15, outcome_col="is_human", feature_cols:list=None, save_path=None, verbose=False):
    '''
    Create X, y from df, split into train, test and val 

    Args: 
        df: dataframe to split
        random_state: seed for split for reproducibility
        val_test_size: size of validation and test sets. 0.15 results in 15% val and 15% test. 
        feature_cols: feature columns in df (predictors)
                      If None, defaults to all viable features (removing outcome column "is_human" and other irrelevant cols)
        outcome_col: column for outcome. Defaults to "is_human"
        verbose: 
        save_path: directory to save splitted data. If None, does not sav

    Returns: 
        splits: dict with all splits 
    '''
    # take all cols for X if feature_cols is unspecified, otherwise subset df to incl. only feature_cols
    if feature_cols == None: 
        cols_to_drop = ["id", "is_human", "dataset", "sample_params", "model", "temperature", "prompt_number", "unique_id"] +  (["annotations"] if "annotations" in df.columns else []) # drop annotation if present (only present for dailydialog)
        X = df.drop(columns=cols_to_drop)
    else:
        X = df[feature_cols]

    # if model col is present, make explicit categorical for xgboost
    if "model" in X.columns:
        X["model"] = X["model"].astype("category")

    # subset df to a single outcome col for y 
    y = df[[outcome_col]]

    splits = {}

    # create train, test, val splits based on val_test_size, save to splits dict. If val_test_size = 0.15, 15% val and 15% test (and stratify by y to keep class balance as much as possible)
    splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"] = train_test_split(X, y, test_size=val_test_size*2, random_state=random_state, stratify=y)

    # split val from test, stratify by y again 
    splits["X_val"], splits["X_test"], splits["y_val"], splits["y_test"] = train_test_split(splits["X_test"], splits["y_test"], test_size=0.5, random_state=random_state, stratify=splits["y_test"])
  
    # validate size of splits, print info msg
    if verbose:
        # make table of split sizes (absolute numbers and percentages)
        split_sizes = pd.DataFrame(index=["train", "val", "test"], columns=["absolute", "percentage"])

        # compute absolute numbers, only for X as they should be the same for y
        total_size = len(df)
        for split in ["train", "val", "test"]:
            split_sizes.loc[split, "percentage"] = len(splits[f"X_{split}"]) / total_size * 100
            split_sizes.loc[split, "absolute"] = len(splits[f"X_{split}"])

        print("\n[INFO:] Split sizes:\n")
        print(split_sizes)
        print(f"\nTotal size: {total_size}")

    # save splits to save_path if specified
    if save_path: 
        save_path.mkdir(parents=True, exist_ok=True)
        # get dataset name from df
        for key, value in splits.items():
            value.to_csv(save_path / f"{key}.csv")

    return splits

def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"

    final_df = load_metrics(
                            human_dir=datapath / "human_metrics", 
                            ai_dir=datapath / "ai_metrics",
                            dataset="stories", temp=1, 
                            human_completions_only=True
                            )


    splits = create_split(final_df, random_state=129, val_test_size=0.15, outcome_col="is_human", verbose=True)
    
    # group by dataset and model
    print("\nPrinting groupby...\n")
    print(final_df.groupby(["dataset", "model"]).size())

    print("\nPrinting class distribution... \n")
    print(f"Y train: {splits['y_train'].value_counts()[0]} AI, {splits['y_train'].value_counts()[1]} human")
    print(f"Y val: {splits['y_val'].value_counts()[0]} AI, {splits['y_val'].value_counts()[1]} human")
    print(f"Y test: {splits['y_test'].value_counts()[0]} AI, {splits['y_test'].value_counts()[1]} human")


if __name__ == "__main__":
    main()