'''
Run PCA for each dataset and temp combination, save results to /results/pca_results.
'''
import argparse
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append(str(pathlib.Path(__file__).parents[3]))
from src.utils.pca import run_PCA, save_PCA_results, get_loadings, plot_loadings, plot_cumulative_variance
from src.utils.process_metrics import load_metrics

def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[3] / "metrics"
    savedir = path.parents[3] / "results" / "classify" / "pca_results"

    # save directories    
    loadingspath = savedir / "loadings"
    cumvarpath = savedir / "cumulative_variance"
    expvarpath = savedir / "explained_variance"
    pcafilepath = savedir / "data"
    objectspath = savedir / "objects"

    for path in [loadingspath, cumvarpath, expvarpath, datapath, objectspath]:
        path.mkdir(parents=True, exist_ok=True)

    # only run for TEMP 1 for now! 
    temp = 1 

    datasets = ["stories", "dailymail_cnn", "mrpc", "dailydialog"]

    for dataset in datasets:
        print(f"[INFO:] Running PCA for {dataset}...")
        
        # get metrics for dataset 
        df = load_metrics(
                            human_dir=datapath / "human_metrics", 
                            ai_dir=datapath / "ai_metrics", 
                            dataset=dataset, 
                            temp=temp, 
                            human_completions_only=True
                            )

        ## DEFINE FEATURES ## 
        # get all possible features
        all_features = df.columns.tolist() 

        # cols that are directly tied to type of generation (and should therefore not be included in classification)
        type_cols = ["model", "id", "is_human", "unique_id", "sample_params", "temperature", "prompt_number", "dataset", "annotations"] 
        
        # cols found by running identify_NA_metrics.py 
        na_cols = ['first_order_coherence', 'second_order_coherence', 'smog', 'pos_prop_SPACE']
        manually_selected_cols = ["pos_prop_PUNCT"]
        
        # final cols to drop 
        cols_to_drop = type_cols + na_cols + manually_selected_cols

        # final feature list after dropping cols
        features = [feat for feat in all_features if feat not in cols_to_drop]

        # run PCA
        pca, pca_df = run_PCA(df, feature_names=features, n_components=len(features), keep_metrics_df=False) # keep_metrics_df=False to only keep pca components and row identifiers
        file_name = f"{dataset}_temp{temp}"

        # cumvar
        plot_cumulative_variance(
                                pca, 
                                f"{dataset.upper()} (Temperature of {temp})", 
                                cumvarpath, 
                                f"{file_name}_CUMVAR.png"
                                )

        # loadings 
        loadings = get_loadings(pca, features)
        components = loadings.columns[:20] # first 20 components 

        for comp in components:
            plot_loadings(loadings, comp, loadingspath / file_name)

        # save results
        pca_df.reset_index().to_csv(pcafilepath / f"{file_name}_data.csv", index=False)
        
        with open(objectspath / f'{file_name}_obj.pkl', 'wb') as file:
            pickle.dump(pca, file)


        with open(expvarpath/ f'{file_name}_EXPVAR.txt', 'w') as file:
            file.write("PRINCIPAL COMPONENTS: EXPLAINED VARIANCE\n")
            file.write(f"Features: {features}\n")

            for i, variance in enumerate(pca.explained_variance_ratio_, start=1):
                file.write(f"pca_{i}: {variance:.8f}\n")                    

        # write expvar as csv 
        expvar_df = pd.DataFrame(pca.explained_variance_ratio_, columns=["explained_variance"])
        expvar_df.to_csv(expvarpath / f"{file_name}_EXPVAR.csv")

if __name__ == "__main__":
    main()