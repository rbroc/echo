'''
Functions to get low-level descriptive metrics (doc length, n_tokens, n_characters, n_sentences) and compute PCA components.
'''
import pathlib 
import pickle

import pandas as pd
import numpy as np

import spacy
import textdescriptives as td

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

def get_descriptive_metrics(df:pd.DataFrame, text_column:str, id_column:str):
    '''
    Extract low level descriptive features doc_length, n_tokens, n_characters and n_sentences 
    '''
    textcol = df[text_column]
    idcol = df[id_column]

    metrics = td.extract_metrics(text=textcol, spacy_model="en_core_web_md", metrics=["descriptive_stats", "quality"])
    subset_metrics = metrics[["doc_length", "n_tokens", "n_characters", "n_sentences"]]
    
    metrics_df = pd.concat([df, subset_metrics], axis=1)
    
    return metrics_df 

def run_PCA(metrics_df:pd.DataFrame, feature_names:list, n_components:int=4):
    '''
    Run PCA on list of feature names. Normalises features prior to running PCA
    '''
    # normalise 
    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(metrics_df[feature_names])

    # run pca 
    pca = PCA(n_components=n_components, random_state=129)
    results = pca.fit_transform(scaled_df)

    # save
    column_names = [f"PC{i}" for i in range(1, n_components + 1)]
    pca_df = pd.DataFrame(data = results)
    pca_df.columns = column_names

    # add new components to overall df 
    df = pd.concat([metrics_df, pca_df],axis=1)

    return pca, df

def save_PCA_results(pca, results_path):
    '''
    Save PCA results to desired path (includes pca obj. and txt file with explained variance)
    '''
    with open(results_path / 'pca_model.pkl', 'wb') as file:
        pickle.dump(pca, file)

    with open(results_path / 'explained_variance.txt', 'w') as file:
        # Write the header
        file.write("PRINCIPAL COMPONENTS: EXPLAINED VARIANCE\n")
        file.write("Original features: 'doc_length', 'n_tokens', 'n_characters', 'n_sentences'\n")

        # Write the PCA components and explained variance
        for i, variance in enumerate(pca.explained_variance_ratio_, start=1):
            file.write(f"pca_{i}: {variance:.8f}\n")

## loadings ## 
def get_loadings(pca, feature_names, n_components):
    '''
    Extract PCA loadings (for investigating which original features explain each component)
    '''
    loadings =  pca.components_.T * np.sqrt(pca.explained_variance_)

    column_names = [f"PC{i}" for i in range(1, n_components + 1)]
    loadings_matrix = pd.DataFrame(loadings, index=feature_names)

    loadings_matrix.columns = column_names

    return loadings_matrix

def plot_loadings(loadings_matrix, component:int=1, outpath=None):
    '''
    Plot PCA loadings (for investigating which original features explain each component)
    '''
    reshaped_matrix = loadings_matrix.reset_index().melt(id_vars="index")

    colors = sns.color_palette()[0]

    plot = sns.catplot(data=reshaped_matrix[reshaped_matrix["variable"]==f"PC{component}"],
                       x = "index", y="value", color=colors, kind='bar')
    plt.xlim()
    plot.set(title=f"PC{component}")

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath / f"PC_{component}.png")
