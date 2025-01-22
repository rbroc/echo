"""
Run XGBOOST classifier for each dataset and temp combination, save results to /results/classify/clf_results on all features.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).parents[2]))
import pickle

from utils.classify import clf_pipeline
from src.utils.cols_to_drop import get_cols_to_drop
from xgboost import XGBClassifier


def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg
    parser.add_argument(
        "-d",
        "--dataset",
        default="dailymail_cnn",
        help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'",
        type=str,
    )
    parser.add_argument(
        "-t", "--temp", default=1, help="Temperature of generations", type=float
    )
    parser.add_argument(
        "-top_n",
        "--top_n_features",
        default=3,
        help="Top N features to include in classification",
        type=int,
    )

    args = parser.parse_args()

    return args


def extract_top_features(feature_importances_df, top_n_features: int = 3):
    """
    Filter feature importances to only include top N features
    """
    print(f"[INFO]: Getting top {top_n_features} features...")
    feature_importances_df = feature_importances_df.sort_values(
        by="importance", ascending=False
    )  # ensure that features are sorted by importance (highest to lowest)
    top_features_df = feature_importances_df.head(top_n_features).drop(
        columns=["index"]
    )  # get top N features with head() after sorting, drop index column

    return top_features_df


def get_transformed_X(df, features, pca_model, scaler):
    X_scaled = scaler.transform(df[features])
    X = pca_model.transform(X_scaled)
    return X


def main():
    args = input_parse()

    dataset, temp = args.dataset, args.temp

    if temp == 1.0:
        temp = int(temp)

    # paths
    path = pathlib.Path(__file__)
    pcapath = (
        path.parents[2] / "results" / "pca_results" / f"temp_{temp}" / "train_model.pkl"
    )
    scalerpath = (
        path.parents[2]
        / "results"
        / "pca_results"
        / f"temp_{temp}"
        / "train_scaler.pkl"
    )

    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    savepath.mkdir(parents=True, exist_ok=True)

    datapath = path.parents[2] / "datasets_complete" / "metrics" / f"temp_{temp}"

    top_n = args.top_n_features

    # filter to only include one dataset
    train_df = pd.read_parquet(datapath / f"train_metrics.parquet")
    val_df = pd.read_parquet(datapath / f"val_metrics.parquet")

    train_df = train_df[train_df["dataset"] == dataset]
    val_df = val_df[val_df["dataset"] == dataset]

    ## DEFINE FEATURES ##
    all_features = train_df.columns.tolist()
    cols_to_drop = get_cols_to_drop()
    features = [feat for feat in all_features if feat not in cols_to_drop]

    # transform data with pca model
    with open(scalerpath, "rb") as file:
        scaler = pickle.load(file)

    with open(pcapath, "rb") as file:
        pca_model = pickle.load(file)

    X_train = get_transformed_X(train_df, features, pca_model, scaler)
    X_val = get_transformed_X(val_df, features, pca_model, scaler)

    y_train = train_df["is_human"].values
    y_val = val_df["is_human"].values

    # get top features
    feature_importances = pd.read_csv(
        savepath
        / "feature_importances"
        / f"{dataset}_temp{temp}"
        / "all_models_all_features.csv"
    )
    top_features_df = extract_top_features(feature_importances, top_n_features=top_n)
    top_features = top_features_df["feature"].tolist()

    # subset to only include top features
    top_indices = [
        int(pc[2:]) - 1 for pc in top_features
    ]  # go from PC1, PC2, PC3 to 0, 1, 2 for indexing
    print(f"[INFO]: Top indices: {top_indices}")
    X_train = X_train[:, top_indices]
    X_val = X_val[:, top_indices]

    # create pc feature names (from shape of X_train)
    pc_feature_names = [f"PC{i}" for i in range(1, X_train.shape[1] + 1)]

    # compute amount of model = human (class 1) versus model != human (class 0) for scale_pos_weight
    human_count = train_df[train_df["model"] == "human"].shape[0]
    non_human_count = train_df[train_df["model"] != "human"].shape[0]
    scale_pos_weight = non_human_count / human_count

    # fit
    clf = XGBClassifier(
        enable_categorical=True,
        use_label_encoder=False,
        random_state=129,
        scale_pos_weight=scale_pos_weight,
    )

    clf, clf_report = clf_pipeline(
        df=train_df,
        clf=clf,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=top_features,
        random_state=129,
        save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}",
        save_filename=f"all_models_top{top_n}_features",
    )


if __name__ == "__main__":
    main()
