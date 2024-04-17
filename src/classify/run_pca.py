import argparse
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.pca import run_PCA, save_PCA_results
from prepare_metrics import load_metrics, filter_metrics

# functions currently not in use!! For plotting loadings 
def plot_cumulative_variance(pca, title, save_dir=None, file_name=None): 
    '''
    plot cumulative explained variance for pca 

    code by https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html 
    '''
    # close previous plot
    plt.close()

    sns.set_theme(rc={'figure.figsize':(10, 10)}) # see https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html 
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title(title)

    if save_dir and file_name:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / file_name)

def create_loadings(pca, feature_names):
    # see stack overflow https://stackoverflow.com/questions/67585809/how-to-map-the-results-of-principal-component-analysis-back-to-the-actual-featur
    PCnames = ['PC'+str(i+1) for i in range(pca.n_components_)]
    loadings = pd.DataFrame(pca.components_, columns=PCnames,index=feature_names)

    return loadings

def plot_loadings(loadings, component:str="PC1", save_dir=None):
    # close previous plot
    plt.close()

    # see stack overflow https://stackoverflow.com/questions/67585809/how-to-map-the-results-of-principal-component-analysis-back-to-the-actual-featur
    sns.set_theme(rc={'figure.figsize':(10, 16)})
    loadings[component].sort_values().plot.barh()

    # add more whitespace
    plt.tight_layout(pad=2)

    if save_dir: 
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{component}.png")

def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"
    savepath = path.parents[0] / "pca_results"
    savepath.mkdir(parents=True, exist_ok=True)

    # only run for TEMP 1 for now! 
    temp = 1 

    # load metrics 
    df = load_metrics(human_dir=datapath / "human_metrics", 
                        ai_dir=datapath / "ai_metrics", temp=temp, human_completions_only=True)

    # filter
    df = filter_metrics(df, percent_NA=0.9, percent_zero=0.9, verbose=False, log_file=path.parents[0] / "pca_results" / "filtered_metrics_log.txt")

    ## DEFINE FEATURES ## 
    # get all possible features
    all_features = df.columns.tolist() 

    # define cols not to include in PCA
    cols_to_not_include = ["model", "id", "is_human", "unique_id", "sample_params", "temperature", "prompt_number", "dataset", "annotations"]

    # remove cols not to include from all features
    features = [feat for feat in all_features if feat not in cols_to_not_include]

    # run PCA
    pca, pca_df = run_PCA(df, feature_names=features, n_components=len(features))

    # save PCA results
    pca_df.to_csv(savepath / f"PCA_data_temp{temp}.csv")


if __name__ == "__main__":
    main()