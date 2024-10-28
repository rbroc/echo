"""
BASELINE: EMBEDDINGS + XGBOOST
Run XGBOOST classifier for each dataset and temp combination on embeddings, save results to /results/classify/clf_results on all features.
"""

import argparse
import pathlib
import sys

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from xgboost import XGBClassifier
from src.utils.classify import clf_pipeline


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

def main():
    args = input_parse()
    path = pathlib.Path(__file__)

    dataset, temp = args.dataset, args.temp

    if temp == 1.0:
        temp = int(temp)

    # load metrics
    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    savepath.mkdir(parents=True, exist_ok=True)

    datapath = path.parents[2] / "datasets_complete" / "text" / f"temp_{temp}"
    embeddingspath = path.parents[2] / "datasets_complete" / "embeddings" / f"temp_{temp}"

    ## a little bit of transformation back and forth, but necessary to be able to filter embeddings by dataset
    # read embeddings (.npy file)
    train_embeddings = np.load(embeddingspath / f"train_embeddings.npy")
    val_embeddings = np.load(embeddingspath / f"val_embeddings.npy")

    # read text data
    train_df = pd.read_parquet(datapath / f"train_text.parquet")
    val_df = pd.read_parquet(datapath / f"val_text.parquet")

    # add embeddings to dataframes
    train_df["embeddings"] = train_embeddings.tolist()
    val_df["embeddings"] = val_embeddings.tolist()

    # filter only to include one dataset
    train_df = train_df[train_df["dataset"] == dataset]
    val_df = val_df[val_df["dataset"] == dataset]

    # get correct format for XGBoost
    X_train_emb = np.array(train_df["embeddings"].tolist())
    X_val_emb = np.array(val_df["embeddings"].tolist())

    y_train = train_df["is_human"].values
    y_val = val_df["is_human"].values

    # compute scale_pos_weight
    human_count = train_df[train_df["model"] == "human"].shape[0]
    non_human_count = train_df[train_df["model"] != "human"].shape[0]
    scale_pos_weight = non_human_count / human_count

    clf = XGBClassifier(
        enable_categorical=True,
        use_label_encoder=False,
        random_state=129,
        scale_pos_weight=scale_pos_weight,
    )

    # fit
    feature_names = ["embeddings", f"dims: {X_train_emb.shape[1]}"]
    clf, clf_report = clf_pipeline(
        df=train_df,
        clf=clf,
        X_train=X_train_emb,
        y_train=y_train,
        X_val=X_val_emb,
        y_val=y_val,
        feature_names=feature_names,
        random_state=129,
        save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}",
        save_filename=f"all_models_embeddings_all_features",
    )

if __name__ == "__main__":
    main()
