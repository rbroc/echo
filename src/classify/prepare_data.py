'''
Prepare data for classifiction
'''
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

def load_metrics(data_dir: pathlib.Path, dataset: str = None, temp:float= None):
    '''
    Load metrics

    Args:
        data_dir: path to directory with metrics
        dataset: name of dataset to load. If None, loads all datasets.
        temp: temperature of generations. If None, loads all temperatures.
    '''
    # define how to load a single file
    def load_file(file):
        df = pd.read_csv(file, index_col=[0])
        if "dataset" not in df.columns:
            df["dataset"] = file.name.split("_")[0]
        return df

    # load multiple files 
    dfs = []
    folder_paths = [path for path in data_dir.iterdir()]

    for folder in sorted(folder_paths, reverse=True):
        print(f"[INFO:] Loading data from {folder.name} ...")
        for file in folder.iterdir():
            if not dataset or dataset in file.name: # if dataset is specified, only load files with that dataset name
                dfs.append(load_file(file))

    # combine all loaded dfs into a single df
    final_df = pd.concat(dfs, ignore_index=True)

    # filter by temperature if specified
    if temp:
        final_df = final_df[final_df["temperature"] == temp]
        # error message if no data found for specified temperature
        if len(final_df) == 0:
            raise ValueError(f"No data found for temperature {temp}")

    # add binary outcome column for classification (human = 1, ai = 0)
    final_df["is_human"] = final_df["model"].apply(lambda x: 1 if x == "human" else 0)

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
        cols_to_drop = ["id", "is_human"] + ["annotations"] if "annotations" in df.columns else [] # drop annotation if present (only present for dailydialog)
        X = df.drop(columns=cols_to_drop)
    else:
        X = df[[feature_cols]]

    # subset df to a single outcome col for y 
    y = df[[outcome_col]]

    splits = {}

    # create train, test, val splits based on val_test_size, save to splits dict. If val_test_size = 0.15, 15% val and 15% test
    splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"] = train_test_split(X, y, test_size=val_test_size*2, random_state=random_state)

    # split val from test
    splits["X_val"], splits["X_test"], splits["y_val"], splits["y_test"] = train_test_split(splits["X_test"], splits["y_test"], test_size=0.5, random_state=random_state)
  
    # validate size of splits, print info msg
    if verbose:
        # make table of split sizes (absolute numbers and percentages)
        split_sizes = pd.DataFrame(index=["train", "val", "test"], columns=["absolute", "percentage"])

        # compute absolute numbers, only for X as they should be the same for y
        total_size = len(df)
        for split in ["train", "val", "test"]:
            split_sizes.loc[split, "percentage"] = len(splits[f"X_{split}"]) / total_size * 100
            split_sizes.loc[split, "absolute"] = len(splits[f"X_{split}"])

        print("[INFO:] Split sizes: \n")
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

    final_df = load_metrics(datapath, dataset="dailydialog", temp=1.5)
    splits = create_split(final_df, random_state=129, val_test_size=0.15, outcome_col="is_human", verbose=True)

    print(final_df["dataset"].unique())
    print(final_df["temperature"].unique())

if __name__ == "__main__":
    main()