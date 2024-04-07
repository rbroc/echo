'''
Build classifiers based on metrics extracted in metrics folder
'''
import pathlib
import pandas as pd

def load_metrics(data_dir:pathlib.Path): 
    '''
    Load metrics
    '''
    folder_paths = [path for path in data_dir.iterdir()]

    dfs = []
    for folder in sorted(folder_paths, reverse=True): # reverse=true, sort in desc order to get human first
        print(f"Loading data frames from {folder.name} ...")
        for file in folder.iterdir():
            df = pd.read_csv(file, index_col=[0]) 
            dfs.append(df)

    # concatenate
    final_df = pd.concat(dfs, ignore_index=True)

    # create is_human col for classification 
    final_df["is_human"] = final_df["model"].apply(lambda x: 1 if x == "human" else 0) 

    return final_df


def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"

    final_df = load_metrics(datapath)
    print(final_df)
    print(final_df[["model", "is_human"]])

if __name__ == "__main__":
    main()