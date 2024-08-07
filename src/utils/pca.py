'''
Functions to compute and plot PCA components.
'''
import pathlib 
import pickle

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

def run_PCA(metrics_df:pd.DataFrame, feature_names:list, n_components:int=4, keep_metrics_df=True):
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
   
    if keep_metrics_df: # add new components to overall metrics df
        df = pd.concat([metrics_df.reset_index(), pca_df.reset_index()], axis=1)
    else: # only keep pca components and identifiers
        metrics_df = metrics_df[["id", "model", "is_human", "temperature", "prompt_number"]]
        df = pd.concat([metrics_df.reset_index(), pca_df.reset_index()], axis=1)

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
def get_loadings(pca, feature_names):
    '''
    Extract PCA loadings (for investigating which original features explain each component)
    
    source: stack overflow https://stackoverflow.com/questions/67585809/how-to-map-the-results-of-principal-component-analysis-back-to-the-actual-featur
    '''
    PCnames = ['PC'+str(i+1) for i in range(pca.n_components_)]
    loadings = pd.DataFrame(pca.components_,columns=PCnames,index=feature_names)

    return loadings

def plot_loadings(loadings, component:str="PC1", save_dir=None):
    '''
    Plot loadings for a given component

    source: stack overflow https://stackoverflow.com/questions/67585809/how-to-map-the-results-of-principal-component-analysis-back-to-the-actual-featur
    '''
    # close previous plot
    plt.close()

    sns.set_theme(rc={'figure.figsize':(10, 16)})
    loadings[component].sort_values().plot.barh()

    # add more whitespace
    plt.tight_layout(pad=2)

    if save_dir: 
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{component}.png")

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