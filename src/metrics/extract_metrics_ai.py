import pandas as pd
import spacy
import textdescriptives as td
from argparse import ArgumentParser
import pathlib 

import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from src.utils.process_generations import get_ai_paths, format_ai_data
from src.utils.get_metrics import get_all_metrics

def input_parse():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", type=str)
    args = parser.parse_args()

    return args

def main(): 
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"

    results_path = path.parents[2] / "results" / "metrics" / "ai_metrics"
    results_path.mkdir(parents=True, exist_ok=True)

    # models
    models = ["beluga7b", "llama2_chat13b", "mistral7b"]

    # load paths
    ai_paths = get_ai_paths(ai_dir=ai_dir, models=models, dataset=args.dataset, temp=1, prompt_numbers=[21, 22])

    # load df
    ai_dfs = [pd.read_json(p, lines=True) for p in ai_paths]

    # combine
    ai_dfs_formatted = format_ai_data(ai_dfs)

    # process
    df = pd.concat(ai_dfs_formatted, ignore_index=True, axis=0)

    # get metrics
    metrics = get_all_metrics(df, "completions", "en_core_web_md")
    
    # save
    metrics.to_csv(results_path / f"{args.dataset}_completions.csv")


if __name__ == "__main__":
    main()