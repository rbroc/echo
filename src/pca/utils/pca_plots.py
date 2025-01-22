'''
Functions to compute and plot PCA components.
'''
import pathlib 
import pickle

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

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
    
    # make title bold and bigger
    plt.title(component, fontsize=20, fontweight='bold')

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