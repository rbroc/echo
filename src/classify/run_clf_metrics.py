import argparse
import pathlib
import sys
import numpy as np
from typing import List

import pandas as pd
from xgboost import XGBClassifier

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from src.utils.classify import clf_pipeline
from src.utils.cols_to_drop import get_cols_to_drop
from utils.split_formatter import MetricsSplitFormatter

def input_parse():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()

def compute_scale_pos_weight(train_df, minority_class: str = "human") -> float:
    human_count = train_df[train_df["model"] == minority_class].shape[0]
    non_human_count = train_df[train_df["model"] != minority_class].shape[0]
    return non_human_count / human_count

def extract_top_features(feature_importances_df, top_n_features: int = 3) -> List:
    print(f"[INFO]: Getting top {top_n_features} features...")
    feature_importances_df = feature_importances_df.sort_values(
        by="importance", ascending=False
    )
    top_pc_features_df = feature_importances_df.head(top_n_features).drop(
        columns=["index"]
    )
    return top_pc_features_df["feature"].tolist()

def filter_X_on_top_features(X: np.ndarray | List[np.ndarray], top_pc_features: list) -> np.ndarray:
    top_indices = [int(pc[2:]) - 1 for pc in top_pc_features]
    print(f"[INFO]: Filtering on top indices: {top_indices} i.e., {top_pc_features}")
    return X[:, top_indices]

def main():
    args = input_parse()
    dataset, temp = args.dataset, args.temp

    if temp == 1.0:
        temp = int(temp)

    # paths
    path = pathlib.Path(__file__)
    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    savepath.mkdir(parents=True, exist_ok=True)
    
    pcapath = path.parents[2] / "results" / "pca_results" / f"temp_{temp}" / "train_model.pkl"
    scalerpath = path.parents[2] / "results" / "pca_results" / f"temp_{temp}" / "train_scaler.pkl"

    data_rootpath = path.parents[2] / "datasets_complete"
    splits_dir = data_rootpath / "metrics" / f"temp_{temp}"

    for feature_selection in ["all_features", "top_features", "all_features_model_stratify"]:
        formatter = MetricsSplitFormatter(splits_dir=splits_dir, dataset=dataset, splits_to_load=["train", "val"])    
        train_df = formatter.get_split("train")
        models = [model for model in train_df["model"].unique() if model != "human"]
                    
        # get features
        all_features = formatter.get_split("train").columns.tolist()
        cols_to_drop = get_cols_to_drop()
        features = [feat for feat in all_features if feat not in cols_to_drop]

        # get data in right format
        X_train, y_train = formatter.get_X_y_data(split_name="train", X_col=features, y_col="is_human")
        X_val, y_val = formatter.get_X_y_data(split_name="val", X_col=features, y_col="is_human")

        # transform X 
        X_train = formatter.transform_X(scalerpath, pcapath, X_train)
        X_val = formatter.transform_X(scalerpath, pcapath, X_val)

        if feature_selection == "top_features":
            feature_importances = pd.read_csv(
                savepath / "feature_importances" / f"{dataset}_temp{temp}" / "all_models_all_features.csv"
            )
            top_features = extract_top_features(feature_importances, args.top_n_features)
            X_train = filter_X_on_top_features(X_train, top_features)
            X_val = filter_X_on_top_features(X_val, top_features)
            feature_names = top_features
            save_filename = f"all_models_top{args.top_n_features}_features"
        else:
            feature_names = [f"PC{i}" for i in range(1, X_train.shape[1] + 1)]
            save_filename = "all_models_all_features"

        # init and train classifier
        clf = XGBClassifier(
            enable_categorical=True,
            use_label_encoder=False,
            random_state=129,
            scale_pos_weight=compute_scale_pos_weight(train_df=formatter.get_split("train")),
        )

        clf, _ = clf_pipeline(
            df=formatter.get_split("train"),
            clf=clf,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
            random_state=129,
            save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}",
            save_filename=save_filename
        )

        if feature_selection == "all_features_model_stratify":
            for model in models:
                print(f"Processing model: {model}")
                formatter = MetricsSplitFormatter(splits_dir=splits_dir, dataset=dataset, splits_to_load=["train", "val"])    
                formatter.filter_split("train", column="model", values=[model, "human"])
                formatter.filter_split("val", column="model", values=[model, "human"])
                save_filename = f"{model}-human_all_features"

                all_features = formatter.get_split("train").columns.tolist()
                cols_to_drop = get_cols_to_drop()
                features = [feat for feat in all_features if feat not in cols_to_drop]

                X_train, y_train = formatter.get_X_y_data(split_name="train", X_col=features, y_col="is_human")
                X_val, y_val = formatter.get_X_y_data(split_name="val", X_col=features, y_col="is_human")

                X_train = formatter.transform_X(scalerpath, pcapath, X_train)
                X_val = formatter.transform_X(scalerpath, pcapath, X_val)

                clf = XGBClassifier(
                    enable_categorical=True,
                    use_label_encoder=False,
                    random_state=129,
                    scale_pos_weight=compute_scale_pos_weight(train_df=formatter.get_split("train")),
                )

                clf, _ = clf_pipeline(
                    df=formatter.get_split("train"),
                    clf=clf,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    feature_names=feature_names,
                    random_state=129,
                    save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}",
                    save_filename=save_filename
                )

if __name__ == "__main__":
    main()