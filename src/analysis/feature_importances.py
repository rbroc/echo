"""
Reverse-engineering the feature importances of a model from XGBOOST importances and PC loadings
"""
import pathlib
import numpy as np
import pandas as pd


DATASETS = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

def main(): 
    temp = 1 
    path = pathlib.Path(__file__)

    dataset = DATASETS[0] # for testing

    importances_path = path.parents[2] / "results" / "classify" / "clf_results" / "feature_importances" / f"{dataset}_temp{temp}" / "all_models_all_features.csv"
    loadings_path = path.parents[2] / "results" / "pca_results" / f"temp_{temp}" / "loadings_matrix.csv"

    # read data
    importances_df = pd.read_csv(importances_path, index_col=[0])
    loadings_df = pd.read_csv(loadings_path, index_col=[0])

    # ensure they match 
    importances_df = importances_df.sort_index()  
    loadings_df = loadings_df.sort_index()        

    # importance values for each PC (1D)
    importances_vector = importances_df["importance"].values

    # extract loadings (2D)
    loadings_matrix = loadings_df.values

    # shapes (loadings_matrix: (n_features, n_PCs), importances_vector: (n_PCs,))
    print(f"Loadings matrix shape: {loadings_matrix.shape}")
    print(f"Importances vector shape: {importances_vector.shape}")

    # multiply each PC's loadings by its importance (dot product)
    feature_importances = np.dot(loadings_matrix, importances_vector)

    # get feature names from the index of loadings_df
    feature_names = loadings_df.index.values

    # create a new dataframe with the feature names and the calculated feature importances
    feature_importances_df = pd.DataFrame({"feature": feature_names, "importance": feature_importances})
    print(feature_importances_df)


if __name__ == "__main__":
    main()
    
