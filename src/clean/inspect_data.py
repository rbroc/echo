import pathlib
import pandas as pd
import json, ndjson 

def load_clean_data(data_rootdir:pathlib.Path, dataset:str="stories"):
    human_path = data_rootdir / dataset / "data.ndjson"
    human_df = pd.read_json(human_path, lines=True)

    return human_df

def load_raw_data(data_rootdir:pathlib.Path=pathlib.Path(__file__).parents[2] / "datasets" / "human_datasets", dataset:str="stories"):
    '''
    load raw data 
    '''
    if dataset == "stories": 
        raw_filepath = data_rootdir / dataset / "stories_5bins_1000tokens_al.json"
        with open(raw_filepath) as f:
            data = json.load(f)
    else: 
        raw_filepath = data_rootdir / dataset / "raw.ndjson"
        with open(raw_filepath) as f:
            data = ndjson.load(f)

    # convert to df 
    df = pd.DataFrame(data)

    return df

def print_n_rows(df, n_rows, col):
    '''print n first rows of a col in df'''
    print(f"[INFO]: Printing '{col}' column")
    for i, row in df.head(n_rows).iterrows():
        col_to_print = row[col]
        print("_____")
        print(col_to_print)

def print_rows(df, row_indices:list=[0, 20], col="source"):
    '''
    print specified rows from list of indices
    '''
    print(f"[INFO]: PRINTING '{col.upper()}' COLUMN")
    for i in row_indices: 
        print("_____\n")
        print(df[col][i])

def inspect_data(data_rootdir:pathlib.Path(), dataset:str="stories", row_indices=[0, 20], inspect_raw=True):
    print(f"[INFO]: PRINTING {dataset.upper()} DATA \n")
    # inspect clean
    print("[INFO]: ______ CLEAN DATA ______")
    df = load_clean_data(data_rootdir, dataset)
    print_rows(df, row_indices, "human_completions")
    print_rows(df, row_indices, "source")

    # inspect raw
    raw_df = load_raw_data(data_rootdir, dataset)
    if inspect_raw:
        print("[INFO]: ______ RAW DATA ______")
        print_rows(raw_df, row_indices, "human_completions")
        print_rows(raw_df, row_indices, "source")

    return df, raw_df

def main(): 
    path = path = pathlib.Path(__file__)
    data_root = path.parents[2] / "datasets" / "human_datasets"

    df, raw_df = inspect_data(data_root)
    

if __name__ == "__main__":
    main()
    

