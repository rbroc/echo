'''
Compute Euclidean Distances
'''
import pathlib
import numpy as np
import pandas as pd

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.distance import compute_distances, compute_distance_human_average

def main():
    np.random.seed(seed=129)

    path = pathlib.Path(__file__)

    analysis_path = path.parents[2] / "results" / "analysis"

    print("[INFO:] READING CSV ...")
    df = pd.read_csv(analysis_path / "PCA_data.csv", index_col=False)

    models = ["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"]
    pc_cols = ["PC1", "PC2", "PC3", "PC4"]

    human_only_df = compute_distance_human_average(df, cols=pc_cols)
    models_df = compute_distances(df, models=models, cols=pc_cols, include_baseline_completions=False)

    # concat 
    result_df = pd.concat([models_df, human_only_df])

    # sort by id and model
    sorted_df = result_df.copy().sort_values(["id", "model"])
    print(sorted_df)

    print("[INFO:] SAVING RESULTS ...")
    sorted_df.to_csv(analysis_path / "distances_PC_cols.csv", index=False)

if __name__ == "__main__":
    main()