import pathlib 
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import spacy
import textdescriptives as td

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(ai_dir, human_datapath, models:list, dataset:str): 
    '''
    Combines dataframes from human and ai generated data.

    Args
        ai_dir (pathlib.Path): Path to ai generated data
        human_datapath (pathlib.Path): Path to human generated data
        models (list): List of models to include in the data
        dataset (str): Name of dataset (dailymail_cnn, mrpc, stories, dailydialog)
    '''

    # paths, access subfolder and its file in ai_dir
    ai_paths = []

    for p in ai_dir.iterdir():
        if p.name in models: # access models folder 
            for f in p.iterdir(): 
                if dataset in f.name: # take only the dataset that is specified from mdl folder
                    ai_paths.append(f)

    ai_paths = sorted(ai_paths)

    # data
    human_df = pd.read_json(human_datapath, lines=True)
    ai_dfs = [pd.read_json(p, lines=True) for p in ai_paths]

    # prepare data for concatenating (similar formatting)
    for idx, df in enumerate(ai_dfs): 
        # subset to only 100 vals (since some have 150 and some have 100)
        new_df = df.loc[:99].copy()
        
        # standardise prompt and completions cols 
        prompt_colname = [col for col in new_df.columns if col.startswith("prompt_")][0] # get column name that starts with prompt_ (e.g., prompt_1, prompt_2, ...)
        new_df["prompt_number"] = prompt_colname.split("_")[1] # extract numbers 1 to 6
        new_df.rename(columns={prompt_colname: "prompt"}, inplace=True)

        mdl_colname = [col for col in new_df.columns if col.endswith("_completions")][0] 
        new_df["model"] = re.sub(r"_completions$", "", mdl_colname)  # remove "_completions" from e.g., "beluga_completions"
        new_df.rename(columns={mdl_colname: "completions"}, inplace=True)

        # replace OG df with new df 
        ai_dfs[idx] = new_df
   
    human_df = human_df.query('id in @ai_dfs[1]["id"]').copy()
    human_df["model"] = "human"
    human_df.drop(["source"], inplace=True, axis=1)
    human_df.rename(columns={"human_completions": "completions"}, inplace=True)

    # add human dfs
    all_dfs = [human_df, *ai_dfs]

    # append human to ai_dfs, concatenate all data
    combined_data = pd.concat(all_dfs, ignore_index=True, axis=0)

    return combined_data

def get_descriptive_metrics(df:pd.DataFrame, text_column:str, id_column:str):
    '''
    Extract low level descriptive features doc_length, n_tokens, n_characters and n_sentences 
    '''
    textcol = df[text_column]
    idcol = df[id_column]

    metrics = td.extract_metrics(text=textcol, spacy_model="en_core_web_lg", metrics=["descriptive_stats", "quality"])
    subset_metrics = metrics[["doc_length", "n_tokens", "n_characters", "n_sentences"]]
    
    metrics_df = pd.concat([df, subset_metrics], axis=1)
    
    return metrics_df 

def run_PCA(metrics_df:pd.DataFrame, feature_names:list, n_components:int=3):
    # normalise 
    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(metrics_df[feature_names])

    # run pca 
    pca = PCA(n_components=n_components)
    results = pca.fit_transform(scaled_df)

    # save
    column_names = [f"PC{i}" for i in range(1, n_components + 1)]
    pca_df = pd.DataFrame(data = results)
    pca_df.columns = column_names

    # add new components to overall df 
    df = pd.concat([metrics_df, pca_df],axis=1)

    return pca, df

def get_loadings(pca, feature_names, n_components):
    loadings =  pca.components_.T * np.sqrt(pca.explained_variance_)

    column_names = [f"PC{i}" for i in range(1, n_components + 1)]
    loadings_matrix = pd.DataFrame(loadings, index=feature_names)

    loadings_matrix.columns = column_names

    return loadings_matrix


def plot_loadings(loadings_matrix, component:int=1, outpath=None):
    reshaped_matrix = loadings_matrix.reset_index().melt(id_vars="index")

    colors = sns.color_palette()[0]

    sns.barplot(data=reshaped_matrix[reshaped_matrix["variable"]==f"PC{component}"], x = "index", y="value", color=colors)

    if outpath:
        plt.savefig(outpath / f"PC_{component}.png")

def plot_loading_scatter(): 
    pass

def load_single_dataset(dataset, models): 
    spacy.util.fix_random_seed(129)
    
    # load data 
    path = pathlib.Path(__file__)

    ai_dir = path.parents[1] / "datasets_ai"
    human_data = path.parents[1] / "datasets" / dataset / "data.ndjson"

    df = load_data(ai_dir, human_data, models, dataset)

    # get metrics, perform PCA 
    metrics_df = get_descriptive_metrics(df, "completions", "id")

    pca, final_df = run_PCA(metrics_df, feature_names=["doc_length", "n_tokens", "n_characters", "n_sentences"])

    print(final_df)
    print(pca.explained_variance_ratio_)

    return pca, final_df

def pipe_all_datasets(datasets, models): 
    spacy.util.fix_random_seed(129)
    
    # load data 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[1] / "datasets_ai"
    plotpath = path.parents[1] / "results" / "PCA_plots"
    plotpath.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    
    for dataset in datasets: 
        human_path = path.parents[1] / "datasets" / dataset / "data.ndjson"
        data = load_data(ai_dir, human_path, models, dataset)
        all_dfs.append(data)

    combined_df = pd.concat(all_dfs, ignore_index=True, axis=0)

    metrics_df = get_descriptive_metrics(combined_df, "completions", "id")

    pca, final_df = run_PCA(metrics_df, feature_names=["doc_length", "n_tokens", "n_characters", "n_sentences"], n_components=4)

    loadings_matrix = get_loadings(pca, feature_names=["doc_length", "n_tokens", "n_characters", "n_sentences"],  n_components=4)

    for component in range(5):
        plot_loadings(loadings_matrix, component, plotpath)

    print(final_df)
    print(pca.explained_variance_ratio_)
    print(loadings_matrix)

    return pca, final_df

def main(): 
    models = ["beluga", "llama2_chat"]
    datasets = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

    pca, final_df = pipe_all_datasets(datasets, models)

if __name__ == "__main__":
    main()