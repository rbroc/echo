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
    n_features = 10

    # create new df 
    metrics = pd.DataFrame()
    
    # add n_features worth of dummy features
    print(f"Extracting {n_features} dummy features not using {spacy_mdl} and not doing it on text column {text_column} ...")
    for i in range(n_features):
        metrics[f"dummy_feature_{i}"] = np.random.randint(0, 1000, size=len(df))

    # add to existing df
    metrics_df = pd.concat([df, metrics], axis=1)

    return metrics_df

def get_ai_metrics(ai_dir, models=["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"], dataset:str="mrpc", temp:int|float=1, save_path=None):
    '''
    Extract metrics for AI completions
    '''
    # load path, only for prompt_numbers 21 (as they are the 2.0 prompts that we settled on, but function is capable of loading whatever you want!)
    ai_paths = get_ai_paths(ai_dir=ai_dir, models=models, dataset=dataset, temp=temp, prompt_numbers=[21]) 

    # load df 
    ai_dfs = [pd.read_json(ai_path, lines=True) for ai_path in ai_paths]

    # format dfs using custom fn
    ai_dfs_formatted = format_ai_data(ai_dfs)

    # concat
    df = pd.concat(ai_dfs_formatted, ignore_index=True, axis=0)

    # drop doc length (as metrics adds it)
    df = df.drop(columns=["doc_length"])

    # extract metrics
    metrics_df = get_metrics_dummy(df, "completions", "en_core_web_md")

    # drop cols 
    metrics_df = metrics_df.drop(columns=["completions", "prompt"])

    # mv model col to front if present in df 
    if "model" in metrics_df.columns: 
        metrics_df.insert(loc=1, column='model', value=metrics_df.pop('model')) # insert mdl col on 2nd position in df  

    if save_path:
        metrics_df.to_csv(save_path)

    return metrics_df

def get_human_metrics(human_dir, dataset:str, save_path=None):
    pass

def main(): 
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"

    metrics_path = path.parents[2] / "metrics" / "ai_metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)


    # define args
    temp = 1
    save_path = metrics_path / f"dummy_{args.dataset}_completions_temp{temp}.csv"

    metrics_df = get_ai_metrics(ai_dir=ai_dir, 
                                models=["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"], 
                                dataset=args.dataset, temp=temp, save_path=save_path
                                )



if __name__ == "__main__":
    main()