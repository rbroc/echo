'''
Compute euclidean distances on a list of features
'''
import pathlib
import numpy as np
import pandas as pd

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[3]))
from src.utils.distance import compute_distances

def main():
    np.random.seed(seed=129)

    path = pathlib.Path(__file__)

    datapath = path.parents[3] / "results" / "prompt_select" / "PCA" / "PCA_data.csv"
    results_path = path.parents[3] / "results" / "prompt_select" / "distance"
    results_path.mkdir(parents=True, exist_ok=True)

    print("[INFO:] READING CSV ...")
    df = pd.read_csv(datapath)
    print(df)

    models = ["beluga7b", "llama2_chat13b"]
    pc_cols = ["PC1", "PC2", "PC3", "PC4"]

    print("[INFO:] COMPUTING DISTANCES ...")
    result_df = compute_distances(df, models, pc_cols)

    print(result_df)
 
    print("[INFO:] SAVING RESULTS ...")
    sorted_df = result_df.copy().sort_values(["id", "model"])
    print(sorted_df)
    
    sorted_df.to_csv(results_path / "distances_all_PC_cols.csv", index=False)


if __name__ == "__main__":
    main()