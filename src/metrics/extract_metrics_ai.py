import pandas as pd
import spacy
import textdescriptives as td
from argparse import ArgumentParser
import pathlib 
import numpy as np
import multiprocessing as mp

import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from src.utils.process_generations import get_ai_paths, format_ai_data
from src.utils.get_metrics import get_all_metrics, get_all_metrics_pipe

def input_parse():
    parser = ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)
    
    # flags to only process either human or ai (e.g., if flag -human only is used, then only human will be processed)
    parser.add_argument("-human_only", default=False, action="store_true") 
    parser.add_argument("-ai_only", default=False, action="store_true")

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

def get_ai_metrics(ai_dir, models=["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"], dataset:str="mrpc", temp:int|float=1, save_dir=None):
    '''
    Extract metrics for AI completions
    '''
    # get paths, only for prompt_numbers 21 (as they are the 2.0 prompts that we settled on, but function is capable of loading whatever you want!)
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

    if save_dir:
        metrics_df.to_csv(save_path / f"{dataset}_completions_temp{temp}.csv")

    return metrics_df

def get_human_metrics(human_dir, dataset:str, batch_size, n_process, save_dir=None):
    '''
    extract metrics for human data 
    '''
    # def paths 
    human_path =  human_dir / dataset / "data.ndjson"

    # load df 
    human_df = pd.read_json(human_path, lines=True)

    # add model col 
    human_df["model"] = "human"

    # process source, completions
    source_df = get_all_metrics_pipe(human_df, text_column="source", batch_size=batch_size, n_process=n_process)
    completions_df = get_all_metrics_pipe(human_df, text_column="source", batch_size=batch_size, n_process=n_process)
    
    # format dfs
    cols_to_drop = ["source", "human_completions"]

    for df in [source_df, completions_df]:
        df.drop(columns=cols_to_drop, inplace=True)
        if "model" in df.columns: 
            df.insert(loc=1, column='model', value=df.pop('model')) # insert mdl col on 2nd position in df  

    if save_dir: 
        source_df.to_csv(save_dir / f"{dataset}_source.csv")
        completions_df.to_csv(save_dir / f"{dataset}_completions.csv")
    
    return source_df, completions_df 


def main(): 
    args = input_parse()
    # define paths 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"
    human_dir = path.parents[2] / "datasets" / "human_datasets"

    metrics_path = path.parents[2] / "metrics" 
    metrics_path.mkdir(parents=True, exist_ok=True)

    # get cores for multiprocessing
    n_cores = mp.cpu_count() - 1

    # HUMAN PROCESSING
    if args.human_only:
        print(f"[INFO:] Processing HUMAN dataset only for '{args.dataset}'")
        source_df, completions_df = get_human_metrics(human_dir=human_dir,
                                    dataset=args.dataset, 
                                    batch_size=20,
                                    n_process=n_cores,
                                    save_dir = metrics_path / "human_metrics"
                                    )

    else: 
        # AI PROCESSING
        for temp in [1, 1.5]:
            print(f"[INFO]: Processing AI datasets only for '{args.dataset}'")
            ai_savefile = metrics_path / "ai_metrics" / f"{args.dataset}_completions_temp{temp}.csv"
        
            ai_metrics_df = get_ai_metrics(ai_dir=ai_dir, 
                                    models=["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"], 
                                    dataset=args.dataset, temp=temp, save_dir= metrics_path / "ai_metrics"
                                    )

        if not args.ai_only: # if args_ai_only not specified, then run human also! 
            print(f"Processing both AI and HUMAN datasets for '{args.dataset}'")
            source_df, completions_df = get_human_metrics(human_dir=human_dir,
                                    dataset=args.dataset, 
                                    batch_size=20,
                                    n_process=n_cores,
                                    save_dir = metrics_path / "human_metrics"
                                    )

if __name__ == "__main__":
    main()