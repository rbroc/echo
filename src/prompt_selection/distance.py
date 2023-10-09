import pathlib
import numpy as np
import pandas as pd

def compute_distances(pca_df:pd.DataFrame, models:list=["beluga", "llama2_chat"], components:list=["PC1", "PC2", "PC3", "PC4"]):
    '''
    Extract euclidean distances between human and model completions in n-dimensions from a list of components. 

    Args
        pca_df: dataframe with PCA components
        models: list of models present in the model column in the dataframe
        components: list of PCA components present in the dataframe

    Returns
        result_df: dataframe containing columns: id, model, dataset, distance, prompt_number
    '''
    result_rows = []    

    # subset pca_df to include only the AI models
    pca_df_ai = pca_df[pca_df['model'].isin(models)]

    for _, row in pca_df_ai.iterrows():
            # extract 'id' for the current row
            current_id = row['id']

            # extract PCs for the 'human' model with the same 'id'
            pc_human = pca_df[(pca_df['model'] == 'human') & (pca_df['id'] == current_id)][components].values

            # extract PCs for model
            pc_model = row[components].values

            # compute euclidean distance in n-dimensions
            distance = np.sqrt(np.sum((pc_human - pc_model) ** 2))

            result_row = {
                'id': row['id'],
                'model': row["model"],
                'dataset': row['id'],
                'distance': distance,
                'prompt_number': row["prompt_number"]
            }

            result_rows.append(result_row)

    result_df = pd.DataFrame(result_rows)

    return result_df

def main():
    path = pathlib.Path(__file__)

    datapath = path.parents[2] / "results" / "PCA" / "PCA_data.csv"

    df = pd.read_csv(datapath)
    
    models = ["beluga", "llama2_chat"]
    pc_components = ['PC1', 'PC2', 'PC3', 'PC4']

    print("[INFO:] COMPUTING DISTANCES ...")
    result_df = compute_distances(df, models, pc_components)

    print(result_df)

if __name__ == "__main__":
    main()