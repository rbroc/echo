'''
Construct classifiers
'''
import pathlib
import argparse
from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics

def clf_fit(classifier, X_train, y_train): 
    '''
    fit an initialized classifier to training data
    '''
    classifier.fit(X_train, y_train)

    return classifier

def clf_evaluate(classifier, X, y):
    '''
    evaluate fitted classifier on data
    '''
    y_pred = classifier.predict(X)

    clf_report = metrics.classification_report(y, y_pred)

    return clf_report

def create_split(df, feature_cols:list, random_state:int=129, val_test_size:float=0.15, outcome_col:str="is_human", save_path:pathlib.Path=None, verbose:bool=False):
    '''
    Create X, y from df, split into train, test and val 

    Args: 
        df: dataframe to split
        random_state: seed for split for reproducibility
        val_test_size: size of validation and test sets. 0.15 results in 15% val and 15% test. 
        feature_cols: feature columns in df (predictors)
        outcome_col: column for outcome. Defaults to "is_human"
        verbose: 
        save_path: directory to save splitted data. If None, does not sav

    Returns: 
        splits: dict with all splits 
    '''
    X = df[feature_cols]

    # if model col is present, make explicit categorical for xgboost
    if "model" in X.columns:
        X["model"] = X["model"].astype("category")

    # subset df to a single outcome col for y 
    y = df[[outcome_col]]

    splits = {}
    # create splits based on val_test_size, save to dict. If val_test_size = 0.15, 15% val and 15% test (stratify by y to somewhat keep class balance)
    splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"] = train_test_split(X, y, test_size=val_test_size*2, random_state=random_state, stratify=y)

    # split val from test, stratify by y again 
    splits["X_val"], splits["X_test"], splits["y_val"], splits["y_test"] = train_test_split(splits["X_test"], splits["y_test"], test_size=0.5, random_state=random_state, stratify=splits["y_test"])
  
    # validate size of splits, print info msg
    if verbose:
        # make table of split sizes (absolute numbers and percentages)
        split_sizes = pd.DataFrame(index=["train", "val", "test"], columns=["absolute", "percentage"])

        # compute absolute numbers, only for X as they should be the same for y
        total_size = len(df)
        for split in ["train", "val", "test"]:
            split_sizes.loc[split, "percentage"] = len(splits[f"X_{split}"]) / total_size * 100
            split_sizes.loc[split, "absolute"] = len(splits[f"X_{split}"])

        print("\n[INFO:] Split sizes:\n")
        print(split_sizes)
        print(f"\nTotal size: {total_size}")

    # save splits to save_path if specified
    if save_path: 
        save_path.mkdir(parents=True, exist_ok=True)
        for key, value in splits.items(): # get dataset name from df
            value.to_csv(save_path / f"{key}.csv")

    return splits

def get_feature_importances(splits, clf):
    # get feature importance, sort by importance
    feature_importances_vals = clf.feature_importances_
    feature_importances = pd.DataFrame({"feature": splits["X_train"].columns, "importance": feature_importances_vals})
    sorted_feature_importances = feature_importances.sort_values(by="importance", ascending=False)

    # return plot 
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 15))
    g = sns.barplot(x="importance", y="feature", data=feature_importances.sort_values(by="importance", ascending=False))
    plt.title("Feature importances")
    
    return sorted_feature_importances

def plot_feature_importances(feature_importances_df, save_dir=None, save_filename:str="feature_importances"):
    # plot 
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 15))
    g = sns.barplot(x="importance", y="feature", data=feature_importances_df, hue="feature", palette="viridis")
    plt.title("Feature importances")

    # save plot
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{save_filename}.png")

def check_splits(splits, df):
    '''
    Print groupby and class distribution for splits

    Args:
        splits: dict with X_train, X_val, X_test, y_train, y_val, y_test
        df: original dataframe from which splits were created
    '''
    # group by dataset and model
    print("\nPrinting groupby...\n")
    print(df.groupby(["dataset", "model"]).size())

    print("\nPrinting class distribution... \n")
    print(f"Y train: {splits['y_train'].value_counts()[0]} AI, {splits['y_train'].value_counts()[1]} human")
    print(f"Y val: {splits['y_val'].value_counts()[0]} AI, {splits['y_val'].value_counts()[1]} human")
    print(f"Y test: {splits['y_test'].value_counts()[0]} AI, {splits['y_test'].value_counts()[1]} human")

def clf_pipeline(df, feature_cols, random_state=129, save_dir=None, save_filename:str="clf_report"): 
    '''
    Pipeline for creating splits, fitting classifier, evaluating classifier and saving evaluation report (on validation data)

    Args:
        df: dataframe to use for classifier
        random_state: seed for reproducibility
        feature_cols: list of features to use for classifier. If None, uses all viable features
        save_dir: directory to save classifier report. If None, does not save
        save_filename: filename for classifier report. Defaults to "clf_report"
    
    Returns:
        splits: dict with X_train, X_val, X_test, y_train, y_val, y_test
        clf: fitted classifier
        clf_report: classification report on validation data
    '''

    # init classifier 
    print("[INFO:] Initializing XGClassifier ...")
    clf = XGBClassifier(enable_categorical=True, use_label_encoder=False, random_state=random_state)

    # creating splits 
    print(f"[INFO:] Creating splits with features: {feature_cols} using random state {random_state} ...")    
    splits = create_split(df, feature_cols=feature_cols, random_state=129, val_test_size=0.15, outcome_col="is_human", verbose=False)

    # fit classifier
    print("[INFO:] Fitting classifier ...")
    clf = clf_fit(clf, splits["X_train"], splits["y_train"])

    # evaluate classifier on val set
    print("[INFO:] Evaluating classifier ...")
    clf_report = clf_evaluate(clf, X=splits["X_val"], y=splits["y_val"])

    # save results 
    if save_dir:
        print("[INFO:] Saving classifier report ...")
        save_dir.mkdir(parents=True, exist_ok=True) # create save dir if it doesn't exist

        # get feature names for report from X_train (as list)
        feature_names = splits["X_train"].columns.tolist()

        with open(f"{save_dir / save_filename}.txt", "w") as file: 
            file.write(f"Results from model run at {datetime.now()}\n")
            file.write(f"Original dataset: {df.dataset.unique()[0]}, temperature: {df.temperature.unique()[1]}\n") # taking second value as temp as 
            file.write(f"Random state: {random_state}\n")
            file.write(f"{clf_report}\n")
            file.write(f"Features: {feature_names}\n")

    return splits, clf, clf_report

def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)
    parser.add_argument("-t", "--temp", default=1, help="Temperature of generations", type=float)

    args = parser.parse_args()

    return args

def main():
    args = input_parse()

    # load data, create splits
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "results" / "classify" / "pca_results" / "data"
    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    savepath.mkdir(parents=True, exist_ok=True)

    dataset, temp = args.dataset, args.temp

    if temp == 1.0:
        temp = int(temp)

    ## ALL FEATURES ## 
    # load data
    df = pd.read_csv(datapath / f"{dataset}_temp{temp}_data.csv")
    df["dataset"] = dataset

    cols = df.columns.tolist()

    pc_features = [col for col in cols if "PC" in col]

    print(pc_features)

    # fit 
    splits, clf, clf_report = clf_pipeline(df, random_state=129, feature_cols=pc_features, save_dir=savepath / "clf_reports", save_filename=f"{dataset}_all_features_temp{temp}")

    # feature importances
    feature_importances = get_feature_importances(splits, clf)
    plot_feature_importances(feature_importances, save_dir=savepath / "feature_importances", save_filename=f"{dataset}_all_features_temp{temp}")

if __name__ == "__main__":
    main()