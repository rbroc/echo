import pathlib 
import re
import pandas as pd

import spacy
import textdescriptives as td

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

    return all_ai_paths, human_paths

def load_data(ai_paths, human_paths):
    pass

def combine_data(ai_dfs, human_dfs):
    pass

def main(): 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[1] / "datasets_ai"
    human_dir = path.parents[1] / "datasets"
    
    models = ["beluga", "llama2_chat"]
    datasets = ["stories", "mrpc"]

    ai_paths, human_paths = get_all_paths(ai_dir, human_dir, models, datasets)
    print(ai_paths)

if __name__ == "__main__":
    main()

