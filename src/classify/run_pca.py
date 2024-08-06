import argparse
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.pca import run_PCA, save_PCA_results
from src.utils.process_metrics import load_metrics

def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"
    savepath = path.parents[0] / "pca_results"
    savepath.mkdir(parents=True, exist_ok=True)

    # only run for TEMP 1 for now! 
    temp = 1 

    datasets = ["stories", "dailymail_cnn", "mrpc", "dailydialog"]

    for dataset in datasets:
        print(f"[INFO:] Running PCA for {dataset}...")
        # load metrics 
        df = load_metrics(human_dir=datapath / "human_metrics", 
                            ai_dir=datapath / "ai_metrics", dataset=dataset, temp=temp, human_completions_only=True)

        ## DEFINE FEATURES ## 
        # get all possible features
        all_features = df.columns.tolist() 

        # define cols not to include in PCA
        type_cols = ["model", "id", "is_human", "unique_id", "sample_params", "temperature", "prompt_number", "dataset", "annotations"] # cols that are directly tied to type of generation
        na_cols = ['contains_lorem ipsum', 'duplicate_line_chr_fraction', 
                    'duplicate_ngram_chr_fraction_10', 'duplicate_ngram_chr_fraction_7', 
                    'duplicate_ngram_chr_fraction_8', 'duplicate_ngram_chr_fraction_9', 
                    'duplicate_paragraph_chr_fraction', 'pos_prop_SYM', 'proportion_bullet_points', 
                    'proportion_ellipsis', 'symbol_to_word_ratio_#', 'first_order_coherence', 'second_order_coherence', 'smog'] # cols found by running identify_NA_metrics.py

        manually_selected_cols = ["pos_prop_SPACE", "pos_prop_PUNCT"]

        # cols to drop
        cols_to_drop = type_cols + na_cols + manually_selected_cols

        # remove cols not to include from all features
        features = [feat for feat in all_features if feat not in cols_to_drop]

        # run PCA
        pca, pca_df = run_PCA(df, feature_names=features, n_components=len(features))

        # save PCA results
        pca_df.to_csv(savepath / f"{dataset}_PCA_temp{temp}.csv")

if __name__ == "__main__":
    main()