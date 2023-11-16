'''
Compute euclidean distances on a list of features
'''
import pathlib
import numpy as np
import pandas as pd

def compute_distances(df:pd.DataFrame, models:list=["beluga", "llama2_chat"], cols:list=["PC1", "PC2", "PC3", "PC4"]):
    '''
    Extract euclidean distances between human and model completions in n-dimensions from a list of features (cols) 

    Args
        df: dataframe with features columns (e.g., PC components)
        models: list of models present in the model column in the dataframe
        cols: list of feature cols present in the dataframe (e.g., PC components)

    Returns
        result_df: dataframe containing columns: id, model, dataset, distance, prompt_number
    '''
    result_rows = []    

    # subset df to include only the AI models
    df_ai = df[df["model"].isin(models)]

    for _, row in df_ai.iterrows():
            # extract "id" for the current row
            current_id = row["id"]

            # extract features for the "human" model with the same "id" as df_ai 
            pc_human = df[(df["model"] == "human") & (df["id"] == current_id)][cols].values

            # extract features for model completions
            pc_model = row[cols].values

            # compute euclidean distance in n-dimensions
            distance = np.sqrt(np.sum((pc_human - pc_model) ** 2))

            result_row = {
                "id": row["id"],
                "model": row["model"],
                "dataset": row["dataset"],
                "distance": distance,
                "prompt_number": row["prompt_number"],
                "completions": row["completions"]
            }

            result_rows.append(result_row)

    result_df = pd.DataFrame(result_rows)

    return result_df

def main():
    np.random.seed(seed=129)

    path = pathlib.Path(__file__)

    datapath = path.parents[2] / "results" / "PCA" / "PCA_data.csv"
    results_path = path.parents[2] / "results" / "distance"
    results_path.mkdir(parents=True, exist_ok=True)

    print("[INFO:] READING CSV ...")
    df = pd.read_csv(datapath)

    models = ["beluga", "llama2_chat"]
    pc_cols = ["PC1", "PC2", "PC3", "PC4"]

    print("[INFO:] COMPUTING DISTANCES ...")
    result_df = compute_distances(df, models, pc_cols)
 
    print("[INFO:] SAVING RESULTS ...")
    sorted_df = result_df.copy().sort_values(["id", "model"])
    print(sorted_df)
    
    sorted_df.to_csv(results_path / "distances_all_PC_cols.csv", index=False)


if __name__ == "__main__":
    main()