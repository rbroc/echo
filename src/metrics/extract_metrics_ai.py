import pandas as pd
import spacy
import textdescriptives as td
from argparse import ArgumentParser
import pathlib 
import numpy as np

import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from src.utils.process_generations import get_ai_paths, format_ai_data
from src.utils.get_metrics import get_all_metrics

def input_parse():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", type=str)
    args = parser.parse_args()

    return args

def get_metrics_dummy(df, text_column:str, spacy_mdl:str="en_core_web_md"):
    '''
    dummy function to test pipeline
    '''
    n_features = 2

    # create new df 
    metrics = pd.DataFrame()
    
    # add n_features worth of dummy features
    print(f"Extracting {n_features} dummy features not using {spacy_mdl} and not doing it on text column {text_column} ...")
    for i in range(n_features):
        metrics[f"dummy_feature_{i}"] = np.random.randint(0, 1000, size=len(df))

    # add to existing df
    metrics_df = pd.concat([df, metrics], axis=1)

    return metrics_df

def main(): 
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"

    metrics_path = path.parents[2] / "metrics" / "ai_metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    # models
    models = ["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"]

    # load paths
    temp = 1 
    ai_paths = get_ai_paths(ai_dir=ai_dir, models=models, dataset=args.dataset, temp=temp, prompt_numbers=[21])

    # load df
    ai_dfs = [pd.read_json(p, lines=True) for p in ai_paths]

    # combine
    ai_dfs_formatted = format_ai_data(ai_dfs)

    # process
    df = pd.concat(ai_dfs_formatted, ignore_index=True, axis=0)

    # get metrics
    metrics = get_metrics_dummy(df, "completions", "en_core_web_md")
    
    # save
    metrics.to_csv(metrics_path / f"dummy_{args.dataset}_completions_temp{temp}.csv")


if __name__ == "__main__":
    main()