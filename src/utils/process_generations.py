'''
Script for preprocessing and combining generations with human data.
'''

import pathlib 
import re
import pandas as pd

def get_paths(ai_dir:pathlib.Path, human_dir:pathlib.Path, models:list, dataset:str):
    '''
    Get all paths pertaining to a particular dataset (e.g., mrpc)
    '''
    # paths, access subfolder and its file in ai_dir
    ai_paths = []

    for p in ai_dir.iterdir():
        if p.name in models: # access models folder 
            for f in p.iterdir(): 
                if dataset in f.name: # take only the dataset that is specified from mdl folder
                    ai_paths.append(f)

    ai_paths = sorted(ai_paths)

    # get human path 
    human_path = human_dir / dataset / "data.ndjson"

    return ai_paths, human_path 

def load_dataset(ai_paths, human_path):
    '''
    Load data from paths extracted from get_paths function
    '''

    ai_dfs = [pd.read_json(p, lines=True) for p in ai_paths]
    human_df = pd.read_json(human_path, lines=True)

    return ai_dfs, human_df

def combine_data(ai_dfs, human_df, subset=None):
    '''
    Return a dataframe for a particular dataset with all AI generations and human data in one.

    Args: 
        ai_dfs: list of dataframes
        human_df: dataframe corresponding to the dfs in ai_dfs 
        subset: whether datasets should be subsetted (subsets ai datasets to n first rows, and subsequently matches the human completions on completion id). For prompt selection, this was set to 99.

    Returns: 
        combined_df: combined dataframe
    '''
    # prepare data for concatenating (similar formatting)
    for idx, df in enumerate(ai_dfs): 
        # subset to only 100 vals (since some have 150 and some have 100)
        if subset:
            new_df = df.loc[:subset].copy()
        else: 
            new_df = df.copy()
        
        # standardise prompt and completions cols 
        prompt_colname = [col for col in new_df.columns if col.startswith("prompt_")][0] # get column name that starts with prompt_ (e.g., prompt_1, prompt_2, ...)
        new_df["prompt_number"] = prompt_colname.split("_")[1] # extract numbers 1 to 6
        new_df.rename(columns={prompt_colname: "prompt"}, inplace=True)

        mdl_colname = [col for col in new_df.columns if col.endswith("_completions")][0] 
        new_df["model"] = re.sub(r"_completions$", "", mdl_colname)  # remove "_completions" from e.g., "beluga_completions"
        new_df.rename(columns={mdl_colname: "completions"}, inplace=True)

        # replace OG df with new df 
        ai_dfs[idx] = new_df
   
    human_df = human_df.query('id in @ai_dfs[1]["id"]').copy()
    human_df["model"] = "human"
    human_df.drop(["source"], inplace=True, axis=1)
    human_df.rename(columns={"human_completions": "completions"}, inplace=True)

    # add human dfs
    all_dfs = [human_df, *ai_dfs]

    # append human to ai_dfs, concatenate all data
    combined_df = pd.concat(all_dfs, ignore_index=True, axis=0)

    return combined_df

def preprocess_datasets(ai_dir, human_dir, models:list, datasets:list, subset=None):
    '''Loads and prepares as many datasets as needed'''

    all_dfs = []

    for dataset in datasets: 
        ai_paths, human_path = get_paths(ai_dir, human_dir, models, dataset)
        ai_dfs, human_df = load_dataset(ai_paths, human_path)
        dataset_df = combine_data(ai_dfs, human_df, subset=subset)
        
        # add dataset col 
        dataset_df["dataset"] = dataset

        all_dfs.append(dataset_df)

    all_dfs_combined = pd.concat(all_dfs, ignore_index=True, axis=0)

    return all_dfs_combined

def main(): 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets_ai"
    human_dir = path.parents[2] / "datasets"
    
    models = ["beluga", "llama2_chat"]
    datasets = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

    all_dfs = preprocess_datasets(ai_dir, human_dir, models, datasets, subset=99)
    
    print(all_dfs)


if __name__ == "__main__":
    main()

