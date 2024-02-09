'''
Inspect data 
'''
import pathlib
import sys
import argparse
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.process_generations import preprocess_datasets

def input_parse():
    parser = argparse.ArgumentParser(description='Inspect data')
    parser.add_argument('--dataset', "-d", type=str, help='dataset to inspect', default="dailydialog")
    args = parser.parse_args()
    return args

def print_example_from_each_model(df, dataset_name, row_indices:list = [20], print_source:bool=False):
    models = df[df["dataset"] == dataset_name]["model"].unique()

    for row_index in row_indices:
        print(f"--------- completion {row_index} --------- \n")
        if print_source:
                source = (df[df["dataset"] == dataset_name]["source"].iloc[row_index])
                print(f"SOURCE:\n {source} \n")

        for model in models:
            example_completion = df[(df["dataset"] == dataset_name) & (df["model"] == model)].reset_index(drop=True)["completions"].iloc[row_index]
            print(f"MODEL: {model.upper()}\n{example_completion}\n")  

def main(): 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"
    human_dir = path.parents[2] / "datasets" / "human_datasets"

    args = input_parse()

    models = ["beluga7b", "llama2_chat13b", "mistral7b"]

    print("[INFO:] Preprocessing datasets ...")
    df = preprocess_datasets(ai_dir = ai_dir, human_dir = human_dir, models=models, datasets=[args.dataset], temp = "temp1")

    # filtered df 
    ds = args.dataset
    indices = [20, 500, 1000, 3600]
    filtered_df = df[(df["prompt_number"] == "21") | (df["model"] == "human") & (df["dataset"] == ds)]
    print_example_from_each_model(filtered_df, ds, row_indices=indices, print_source=True)
    
if __name__ == "__main__":
    main()