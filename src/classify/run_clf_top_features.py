'''
Run XGBOOST classifier for each dataset and temp combination, save results to /results/classify/clf_results on all features.
'''
import pathlib
import argparse
import pandas as pd
import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.classify import clf_pipeline

def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)
    parser.add_argument("-t", "--temp", default=1, help="Temperature of generations", type=float)
    parser.add_argument("-top_n", "--top_n_features", default=3, help="Top N features to include in classification", type=int)

    args = parser.parse_args()

    return args

def extract_top_features(feature_importances_df, top_n_features:int=3):
    '''
    Filter feature importances to only include top N features
    '''
    print(f"[INFO]: Getting top {top_n_features} features...")
    feature_importances_df = feature_importances_df.sort_values(by="importance", ascending=False) # ensure that features are sorted by importance (highest to lowest)
    top_features_df = feature_importances_df.head(top_n_features).drop(columns=["index"]) # get top N features with head() after sorting, drop index column

    return top_features_df


def main():
    args = input_parse()

    # paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "results" / "classify" / "pca_results" / "data"
    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    savepath.mkdir(parents=True, exist_ok=True)
    
    # load data
    dataset, temp, top_n = args.dataset, args.temp, args.top_n_features

    if temp == 1.0:
        temp = int(temp)

    # read in df that will train (on the top N components)
    df = pd.read_csv(datapath / f"{dataset}_temp{temp}_data.csv")
    df["dataset"] = dataset

    # read in feature importances, filter cols to only include N most important PC comps 
    feature_importances = pd.read_csv(savepath / "feature_importances" / f"{dataset}_temp{temp}" / "all_models_all_features.csv")
    top_features_df = extract_top_features(feature_importances, top_n_features=top_n)
    top_features = top_features_df["feature"].tolist()

    # fit 
    splits, clf, clf_report = clf_pipeline(df, random_state=129, feature_cols=top_features, save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}", save_filename=f"all_models_top{top_n}_features")

if __name__ == "__main__":
    main()