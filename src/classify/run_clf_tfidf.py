"""
Run TFIDF classifier for each dataset and temp combination, save results to /results/classify/clf_results on all features.
"""

import argparse
import pathlib
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from sklearn.linear_model import LogisticRegression
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


def vectorise(X_train, X_val, **kwargs):
    # init vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=False, **kwargs
    )  # lowercase = False as text is already preprocessed

    # fit vectorizer on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # transform validation and test data
    X_val_tfidf = vectorizer.transform(X_val)

    # get feature names
    feature_names = vectorizer.get_feature_names_out()
    print(f"[INFO]: Number of features: {len(feature_names)}")

    return X_train_tfidf, X_val_tfidf, feature_names


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

    # filter to only include one dataset
    train_df = pd.read_parquet(datapath / f"train_text.parquet")
    val_df = pd.read_parquet(datapath / f"val_text.parquet")

    train_df = train_df[train_df["dataset"] == dataset]
    val_df = val_df[val_df["dataset"] == dataset]

    # vectorise data
    max_features_tfidf = 1000

    X_train_tfidf, X_val_tfidf, feature_names = vectorise(
        X_train=train_df["completions"].tolist(),
        X_val=val_df["completions"].tolist(),
        max_features=max_features_tfidf,
    )

    y_train = train_df["is_human"].values
    y_val = val_df["is_human"].values

    clf = LogisticRegression(random_state=129)

    # fit
    clf, clf_report = clf_pipeline(
        df=train_df,
        clf=clf,
        X_train=X_train_tfidf,
        y_train=y_train,
        X_val=X_val_tfidf,
        y_val=y_val,
        feature_names=feature_names,
        random_state=129,
        save_dir=savepath / "clf_reports" / f"{dataset}_temp{temp}",
        save_filename=f"all_models_tfidf_{max_features_tfidf}_features",
    )


if __name__ == "__main__":
    main()
