import pandas as pd
#import spacy
#import textdescriptives as td
import pathlib 
import re

def load_data(ai_dir, human_data, models:list, dataset:str): 
    '''
    Loads data and combines dataframes from human and ai generated data.

    Args
        ai_dir (pathlib.Path): Path to ai generated data
        human_data (pathlib.Path): Path to human generated data
        models (list): List of models to include in the data
        dataset (str): Name of dataset (dailymail_cnn, mrpc, stories, dailydialog)
    '''

    # access subfolder and its file in ai_dir
    ai_paths = []

    for p in ai_dir.iterdir():
        if p.name in models: 
            for f in p.iterdir():
                if dataset in f.name:
                    ai_paths.append(f)

    ai_paths = sorted(ai_paths)

    # load data
    data = pd.read_json(human_data, lines=True)
    dfs = [pd.read_json(p, lines=True) for p in ai_paths]

    for df in dfs: 
        # fix prompt col
        prompt_colname = [col for col in df.columns if col.startswith("prompt_")][0] # get column name that starts with prompt_ (e.g., prompt_1, prompt_2, ...)
        df["prompt_number"] = prompt_colname.split("_")[1]
        df.rename(columns={prompt_colname: "prompt"}, inplace=True)

        # fix model completions col
        mdl_colname = [col for col in df.columns if col.endswith("_completions")][0] 
        df["model"] = re.sub(r"_completions$", "", mdl_colname)  # Use regex to remove "_completions"

        # drop mdl_colname
        df.drop(columns=[mdl_colname], inplace=True)

    # concat dataframes
    ai_combined = pd.concat(dfs, ignore_index=True)

    return ai_combined

def main(): 
    path = pathlib.Path(__file__)
    models = ["beluga", "llama2_chat"]
    dataset = "dailymail_cnn"

    ai_dir = path.parents[1] / "datasets_ai"
    human_data = path.parents[1] / "datasets" / dataset / "data.ndjson"

    ai_combined = load_data(ai_dir, human_data, models, dataset)
    print(ai_combined)


if __name__ == "__main__":
    main()