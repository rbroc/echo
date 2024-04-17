import pathlib
from prepare_data import load_metrics

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[2] / "src"))
from utils.process_generations import preprocess_datasets
from tqdm import tqdm

def main(): 
    path = pathlib.Path(__file__)

    # load completions 
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"
    human_dir = path.parents[2] / "datasets" / "human_datasets"

    models = ["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"]
    datasets = ["stories", "dailymail_cnn", "mrpc", "dailydialog"]

    completion_df = preprocess_datasets(ai_dir, human_dir, models, datasets, subset=None, temp=1, prompt_numbers=[21])

    # load metrics
    metrics_dir = path.parents[2] / "metrics" 

    metrics_df = load_metrics(human_dir=metrics_dir / "human_metrics", 
                                        ai_dir=metrics_dir / "ai_metrics", temp=1, 
                                        human_completions_only=True
                )
    
    # merge
    completion_df = completion_df[["id", 'model', 'completions']]

    #print("[INFO]: Merging completions and metrics...")
    merged_df = completion_df.merge(metrics_df, on=["id", "model"], how="left")

    print(merged_df[["id", "alpha_ratio", "completions", "model"]])

if __name__ == "__main__":
    main()