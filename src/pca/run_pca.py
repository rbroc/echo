"""
Run PCA on train data (metrics) and save results to /results/classify/pca_results/temp_{temp}
"""

import pathlib
import pickle
import sys

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.cols_to_drop import get_cols_to_drop
from src.utils.pca_plots import (get_loadings, plot_cumulative_variance,
                                 plot_loadings)


def run_PCA(df: pd.DataFrame, feature_names: list, random_state:int=129, n_components=None):
    '''
    Run PCA on a list of feature names. Normalises features prior to running PCA.
    
    Args:
        df: dataframe to scale and run PCA on.
        feature_names: List of column names to use for PCA.
        random_state: Random state for PCA. Default is 129.
        n_components: Number of principal components to keep. Default is None (keep all components).
        
    Returns:
        pca: The fitted PCA model.
        scaler: The fitted StandardScaler object for future scaling.
    '''
    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(df[feature_names]) # fit and transform on train data for PCA

    pca_model = PCA(n_components=n_components, random_state=random_state) 
    pca_model.fit(scaled_df) # only fit since we will transform train and test data with the model when we use it

    return pca_model, std_scaler

def main():
    path = pathlib.Path(__file__)

    temp = 1

    # dirs based on temp
    savedir = path.parents[2] / "results" / "pca_results" / f"temp_{temp}"
    loadingspath = savedir / "loadings"

    datapath = (
        path.parents[2]
        / "datasets_complete"
        / "metrics"
        / f"temp_{temp}"
        / "train_metrics.parquet"
    )

    file_name = "train"

    # load train metrics data
    train_df = pd.read_parquet(datapath)

    # DEFINE FEATURES #
    all_features = train_df.columns.tolist()

    # cols to drop
    cols_to_drop = get_cols_to_drop() # see dropped cols in src/utils/cols_to_drop.py

    # final feature list after dropping cols
    features = [feat for feat in all_features if feat not in cols_to_drop]

    # run PCA
    print(f"[INFO:] Running PCA")
    pca_model, scaler = run_PCA(
        train_df,
        feature_names=features,
        random_state=129,
    )

    # cumvar
    plot_cumulative_variance(
        pca_model,
        f"{file_name.capitalize()} data (Temperature of {temp})",
        savedir,
        f"{file_name}_CUMVAR.png",
    )

    # loadings
    loadings = get_loadings(pca_model, features)
    components = loadings.columns.tolist()
    loadings.to_csv(savedir / "loadings_matrix.csv")

    for comp in tqdm(components, desc="[INFO:] Plotting loadings"):
        plot_loadings(loadings, comp, loadingspath)

    # save results + model
    print(f"[INFO]: Saving results")
    with open(savedir / f"{file_name}_model.pkl", "wb") as file:
        pickle.dump(pca_model, file)

    with open(savedir / f"{file_name}_scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    with open(savedir / f"{file_name}_EXPVAR.txt", "w") as file:
        file.write("PRINCIPAL COMPONENTS: EXPLAINED VARIANCE\n")
        file.write(f"Features: {features}\n")

        for i, variance in enumerate(pca_model.explained_variance_ratio_, start=1):
            file.write(f"pca_{i}: {variance:.8f}\n")

        expvar_df = pd.DataFrame(
            pca_model.explained_variance_ratio_, columns=["explained_variance"]
        )
        expvar_df.to_csv(savedir / f"{file_name}_EXPVAR.csv")


if __name__ == "__main__":
    main()
