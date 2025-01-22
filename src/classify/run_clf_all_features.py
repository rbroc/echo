"""
Run XGBOOST classifier for each dataset and temp combination, save results to /results/classify/clf_results on all features.
"""

import argparse
import pathlib
import pickle
import sys

import pandas as pd
from xgboost import XGBClassifier

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from utils.classify import (
    clf_pipeline,
    get_feature_importances,
    plot_feature_importances,
)
from src.utils.cols_to_drop import get_cols_to_drop


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

    args = parser.parse_args()

    return args


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
        feature_names=pc_feature_names,
        random_state=129,
        save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}",
        save_filename=f"all_models_all_features",
    )

    # get feature importances
    feature_importances = get_feature_importances(pc_feature_names, clf)
    plot_feature_importances(
        feature_importances,
        save_dir=savepath / "feature_importances" / f"{dataset}_temp{temp}",
        save_filename=f"all_models_all_features",
    )

    feature_importances.reset_index().to_csv(
        savepath
        / "feature_importances"
        / f"{dataset}_temp{temp}"
        / "all_models_all_features.csv",
        index=False,
    )  # save feature importances to csv (to load in top_features)

    ## ALL FEATURES, SINGLE MODEL ##
    models = [model for model in train_df["model"].unique() if model != "human"]

    for model in models:
        train_df_mdl = train_df[
            (train_df["model"] == model) | (train_df["model"] == "human")
        ]  # subset to particular model and human
        val_df_mdl = val_df[(val_df["model"] == model) | (val_df["model"] == "human")]

        y_train_mdl = train_df_mdl["is_human"].values
        y_val_mdl = val_df_mdl["is_human"].values

        # get transformed X with pca model
        X_train_mdl = get_transformed_X(train_df_mdl, features, pca_model, scaler)
        X_val_mdl = get_transformed_X(val_df_mdl, features, pca_model, scaler)

        # get scale pos weight for good measure (should be 1 as now we are comparing one model to human)
        human_count = train_df_mdl[train_df_mdl["model"] == "human"].shape[0]
        non_human_count = train_df_mdl[train_df_mdl["model"] != "human"].shape[0]
        scale_pos_weight = non_human_count / human_count

        clf = XGBClassifier(
            enable_categorical=True,
            use_label_encoder=False,
            random_state=129,
            scale_pos_weight=scale_pos_weight,
        )

        clf, clf_report = clf_pipeline(
            df=train_df_mdl,  # only used for metadata
            clf=clf,
            X_train=X_train_mdl,
            y_train=y_train_mdl,
            X_val=X_val_mdl,
            y_val=y_val_mdl,
            feature_names=pc_feature_names,
            random_state=129,
            save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}",
            save_filename=f"{model}-human_all_features",
        )

        feature_importances = get_feature_importances(pc_feature_names, clf)
        plot_feature_importances(
            feature_importances,
            save_dir=savepath / "feature_importances" / f"{dataset}_temp{temp}",
            save_filename=f"{model}-human_all_features",
        )


if __name__ == "__main__":
    main()
