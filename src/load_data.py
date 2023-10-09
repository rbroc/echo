import pathlib 
import re
import pandas as pd

def get_paths(ai_dir:pathlib.Path, human_dir:pathlib.Path, models:list, dataset:str):
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


def get_all_paths(ai_dir:pathlib.Path, human_dir:pathlib.Path, models:list, datasets:list):
    all_ai_paths = []
    human_paths = []

    for dataset in datasets: 
        ai_paths, human_path = get_paths(ai_dir, human_dir, models, dataset)
        all_ai_paths.extend(ai_paths)
        human_paths.append(human_path)

    human_paths = sorted(human_paths)

    return all_ai_paths, human_paths

def load_data(ai_paths, human_paths):
    ai_dfs = [pd.read_json(p, lines=True) for p in ai_paths]
    human_dfs = [pd.read_json(p, lines=True) for p in human_paths]

    return ai_dfs, human_dfs

def combine_data(ai_dfs, human_dfs):
    # prepare data for concatenating (similar formatting)
    for idx, df in enumerate(ai_dfs): 
        # subset to only 100 vals (since some have 150 and some have 100)
        new_df = df.loc[:99].copy()
        
        # standardise prompt and completions cols 
        prompt_colname = [col for col in new_df.columns if col.startswith("prompt_")][0] # get column name that starts with prompt_ (e.g., prompt_1, prompt_2, ...)
        new_df["prompt_number"] = prompt_colname.split("_")[1] # extract numbers 1 to 6
        new_df.rename(columns={prompt_colname: "prompt"}, inplace=True)

        mdl_colname = [col for col in new_df.columns if col.endswith("_completions")][0] 
        new_df["model"] = re.sub(r"_completions$", "", mdl_colname)  # remove "_completions" from e.g., "beluga_completions"
        new_df.rename(columns={mdl_colname: "completions"}, inplace=True)

        # replace OG df with new df 
        ai_dfs[idx] = new_df

        for i in range (len(ai_dfs)):
            for idx, df in enumerate(human_dfs):
                human_df = df.query(f'id in @ai_dfs[{i}]["id"]').copy()
                print(human_df)

    for idx, df in enumerate(human_dfs):
        new_df = df
        new_df["model"] = "human"
        new_df.drop(["source"], inplace=True, axis=1)
        new_df.rename(columns={"human_completions": "completions"}, inplace=True)
        human_dfs[idx] = new_df

    # combine all data 
    all_dfs = [*human_dfs, *ai_dfs]
    combined_df = pd.concat(all_dfs, ignore_index=True, axis=0)

    return combined_df

def main(): 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[1] / "datasets_ai"
    human_dir = path.parents[1] / "datasets"
    
    models = ["beluga", "llama2_chat"]
    datasets = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

    ai_paths, human_paths = get_all_paths(ai_dir, human_dir, models, datasets)

    ai_dfs, human_dfs = load_data(ai_paths, human_paths)

    df = combine_data(ai_dfs, human_dfs) 
    
    #print(df)


if __name__ == "__main__":
    main()

