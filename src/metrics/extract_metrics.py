'''
Script to extract metrics for human and AI-generated datasets. 
See metrics/README.md for instructions on how to run.
'''
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
from src.utils.get_metrics import get_all_metrics_pipe, get_information_metrics

def input_parse():
    parser = ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", 
                        help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)
    
    # flags to only process either human or ai (e.g., if flag -human only is used, then only human will be processed)
    parser.add_argument("-human_only", default=False, action="store_true") 
    parser.add_argument("-ai_only", default=False, action="store_true")

    args = parser.parse_args()

    return args

def get_ai_metrics(
                    ai_dir, 
                    models=["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"], 
                    dataset:str="mrpc", 
                    temp:int|float=1, 
                    batch_size:int=1, 
                    n_process:int=1, 
                    compute_perplexity:bool=True,
                    save_dir=None
    ):
    '''
    Extract metrics for AI completions

    Args:
        ai_dir: path to directory with AI completions
        models: list of models to extract metrics for
        dataset: name of dataset to extract metrics for
        temp: temperature of completions
        batch_size: batch size for processing
        n_process: number of processes to use for multiprocessing
        compute_perplexity: whether to compute perplexity and entropy manually with GPT-2 (True) or relying on textdescriptives (False)
        save_dir: path to save directory. If None, does not save.

    Returns:
        completions_df: dataframe with metrics for completions
    '''
    # get paths, only for prompt_numbers 21 (as they are the 2.0 prompts that we settled on, but fn is capable of loading whatever you want!)
    ai_paths = get_ai_paths(ai_dir=ai_dir, models=models, dataset=dataset, temp=temp, prompt_numbers=[21]) 
    ai_dfs = [pd.read_json(ai_path, lines=True) for ai_path in ai_paths]

    # format dfs using custom fn 
    ai_dfs_formatted = format_ai_data(ai_dfs, clean_ai = True)

    # concat
    ai_df = pd.concat(ai_dfs_formatted, ignore_index=True, axis=0)

    # drop doc length (as metrics adds it, and will get confused when it has two cols that are duplicate)
    ai_df = ai_df.drop(columns=["doc_length"])

    # extract metrics
    completions_df = get_all_metrics_pipe(
                                            ai_df, 
                                            text_column="completions", 
                                            batch_size=batch_size, 
                                            n_process=n_process
                                            )
    
    if compute_perplexity:
        completions_df = get_information_metrics(completions_df, text_column = "completions", model_id="gpt2", batch_size=batch_size)
    
    # drop cols 
    completions_df = completions_df.drop(columns=["completions", "prompt"])

    # mv model col to front if present in df 
    if "model" in completions_df.columns: 
        completions_df.insert(loc=1, column='model', value=completions_df.pop('model')) # insert mdl col on 2nd position in df  

    if save_dir:
        completions_df.to_csv(save_dir / f"{dataset}_completions_temp{temp}.csv")

    return completions_df

def get_human_metrics(
                        human_dir, 
                        dataset:str="mrpc", 
                        batch_size:int=1, 
                        n_process:int=1, 
                        compute_perplexity:bool=True,
                        save_dir=None
    ):
    '''
    extract metrics for human data 

    Args:
        human_dir: path to directory with human data
        dataset: name of dataset to extract metrics for
        batch_size: batch size for processing
        n_process: number of processes to use for multiprocessing
        compute_perplexity: whether to compute perplexity and entropy manually with GPT-2 (True) or relying on textdescriptives (False)
        save_dir: path to save directory. If None, does not save.

    Returns:
        source_df: dataframe with metrics for source text
        completions_df: dataframe with metrics for completions
    '''
    # def paths 
    human_path =  human_dir / dataset / "data.ndjson"

    # load df 
    human_df = pd.read_json(human_path, lines=True)

    # add model col 
    human_df["model"] = "human"

    # process source, completions
    source_df = get_all_metrics_pipe(human_df, text_column="source", batch_size=batch_size, n_process=n_process)
    completions_df = get_all_metrics_pipe(human_df, text_column="human_completions", batch_size=batch_size, n_process=n_process)

    # compute perplexity and entropy manually
    if compute_perplexity:
        source_df = get_information_metrics(source_df, text_column = "source", model_id="gpt2", batch_size=batch_size)
        completions_df = get_information_metrics(completions_df, text_column = "human_completions", model_id="gpt2", batch_size=batch_size)
    
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

    # get cores for multiprocessing (-1 for safety)
    n_cores = mp.cpu_count() - 1
    batch_size = 100

    # HUMAN PROCESSING
    if args.human_only:
        print(f"[INFO:] Processing HUMAN dataset for '{args.dataset}'")
        source_df, completions_df = get_human_metrics(human_dir=human_dir,
                                    dataset=args.dataset, 
                                    batch_size=batch_size,
                                    n_process=n_cores,
                                    compute_perplexity=False, # for now until we decide on a model
                                    save_dir = metrics_path / "human_metrics"
                                    )

    else: 
        # AI PROCESSING
        for temp in [1, 1.5]:
            print(f"[INFO]: Processing AI datasets for '{args.dataset}'")
            ai_savefile = metrics_path / "ai_metrics" / f"{args.dataset}_completions_temp{temp}.csv"
        
            ai_metrics_df = get_ai_metrics(ai_dir=ai_dir, 
                                    models=["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"], 
                                    dataset=args.dataset, 
                                    temp=temp, 
                                    batch_size=batch_size, 
                                    compute_perplexity=False, # for now until we decide on a model
                                    n_process=n_cores,
                                    save_dir= metrics_path / "ai_metrics"
                                    )

        if not args.ai_only: # if args_ai_only not specified, then run human also! 
            print(f"Processing HUMAN datasets for '{args.dataset}'")
            source_df, completions_df = get_human_metrics(human_dir=human_dir,
                                    dataset=args.dataset, 
                                    batch_size=20,
                                    n_process=n_cores,
                                    compute_perplexity=False, # for now until we decide on a model
                                    save_dir = metrics_path / "human_metrics"
                                    )

if __name__ == "__main__":
    main()