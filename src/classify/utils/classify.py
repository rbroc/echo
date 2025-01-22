"""
functions for classification tasks (e.g. fitting classifier, evaluating classifier, creating splits, plotting feature importances). 
Currently only used in src/classify
"""

import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics


def clf_pipeline(
    df,
    clf,
    X_train,
    y_train,
    X_val,
    y_val,
    feature_names: list,
    random_state=129,
    save_dir=None,
    save_filename: str = "clf_report",
):
    """
    Pipeline for fitting instantiated classifier, evaluating classifier and saving evaluation report (on validation data)

    Args:
        df: dataframe that is trained on (for metadata)
        clf: instantiated classifier
        random_state: seed for reproducibility
        save_dir: directory to save classifier report. If None, does not save
        save_filename: filename for classifier report. Defaults to "clf_report"

    Returns:
        clf: fitted classifier
        clf_report: classification report on validation data
    """
    # fit classifier
    print("[INFO:] Fitting classifier ...")
    clf.fit(X_train, y_train)

    # evaluate classifier on val set
    print("[INFO:] Evaluating classifier ...")
    y_pred = clf.predict(X_val)
    clf_report = metrics.classification_report(y_val, y_pred)

    # save results
    if save_dir:
        print("[INFO:] Saving classifier report ...")
        save_dir.mkdir(
            parents=True, exist_ok=True
        )  # create save dir if it doesn't exist

        with open(f"{save_dir / save_filename}.txt", "w") as file:
            file.write(f"Results from model run at {datetime.now()}\n")
            file.write(
                f"Original dataset: {df.dataset.unique()[0]}, temperature: {df.temperature.unique()[1]}\n"
            )  # taking second value as temp as
            file.write(f"Random state: {random_state}\n")
            file.write(f"{clf_report}\n")
            file.write(
                f"Model(s) compared with human:{[model for model in df['model'].unique() if model != 'human']}\n"
            )
            file.write(f"Features: {feature_names}\n")

    return clf, clf_report


def get_feature_importances(feature_names, clf):
    # get feature importance, sort by importance
    feature_importances_vals = clf.feature_importances_
    feature_importances = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances_vals}
    )
    sorted_feature_importances = feature_importances.sort_values(
        by="importance", ascending=False
    )

    # return plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 15))
    g = sns.barplot(
        x="importance",
        y="feature",
        data=feature_importances.sort_values(by="importance", ascending=False),
    )
    plt.title("Feature importances")

    return sorted_feature_importances


def plot_feature_importances(
    feature_importances_df, save_dir=None, save_filename: str = "feature_importances"
):
    # plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 15))
    g = sns.barplot(
        x="importance",
        y="feature",
        data=feature_importances_df,
        hue="feature",
        palette="viridis",
    )
    plt.title("Feature importances")

    # save plot
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{save_filename}.png")
