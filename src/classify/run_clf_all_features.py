'''
Run XGBOOST classifier for each dataset and temp combination, save results to /results/classify/clf_results on all features.
'''
import pathlib
import argparse
import pandas as pd
import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.classify import clf_pipeline, get_feature_importances, plot_feature_importances

def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)
    parser.add_argument("-t", "--temp", default=1, help="Temperature of generations", type=float)

    args = parser.parse_args()

    return args

def main():
    args = input_parse()

    # paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "results" / "classify" / "pca_results" / "data"
    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    savepath.mkdir(parents=True, exist_ok=True)
    
    # load data
    dataset, temp = args.dataset, args.temp

    if temp == 1.0:
        temp = int(temp)

    df = pd.read_csv(datapath / f"{dataset}_temp{temp}_data.csv")
    df["dataset"] = dataset

    ## ALL FEATURES, ALL MODELS ## 
    cols = df.columns.tolist()
    pc_features = [col for col in cols if "PC" in col]

    # fit 
    splits, clf, clf_report = clf_pipeline(df, random_state=129, feature_cols=pc_features, save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}", save_filename=f"all_models_all_features")

    # get feature importances
    feature_importances = get_feature_importances(splits, clf)
    plot_feature_importances(feature_importances, save_dir=savepath / "feature_importances" / f"{dataset}_temp{temp}", save_filename=f"all_models_all_features")
    
    feature_importances.reset_index().to_csv(savepath / "feature_importances" / f"{dataset}_temp{temp}" / "all_models_all_features.csv", index=False) # save feature importances to csv (to load in top_features)
    
    ## ALL FEATURES, SINGLE MODEL ##
    models = [model for model in df["model"].unique() if model != "human"]
    
    for model in models:
        model_df = df[(df["model"] == model) | (df["model"] == "human")] # subset to particular model and human
        splits, clf, clf_report = clf_pipeline(model_df, random_state=129, feature_cols=pc_features, save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}", save_filename=f"{model}-human_all_features")

        feature_importances = get_feature_importances(splits, clf)
        plot_feature_importances(feature_importances, save_dir=savepath / "feature_importances" / f"{dataset}_temp{temp}", save_filename=f"{model}-human_all_features")

if __name__ == "__main__":
    main()