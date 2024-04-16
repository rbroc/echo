'''
Construct classifiers
'''
import pathlib
import argparse
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics

from prepare_data import load_metrics, filter_metrics

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

def create_split(df, random_state=129, val_test_size:float=0.15, outcome_col="is_human", feature_cols:list=None, save_path=None, verbose=False):
    '''
    Create X, y from df, split into train, test and val 

    Args: 
        df: dataframe to split
        random_state: seed for split for reproducibility
        val_test_size: size of validation and test sets. 0.15 results in 15% val and 15% test. 
        feature_cols: feature columns in df (predictors)
                      If None, defaults to all viable features (removing outcome column "is_human" and other irrelevant cols)
        outcome_col: column for outcome. Defaults to "is_human"
        verbose: 
        save_path: directory to save splitted data. If None, does not sav

    Returns: 
        splits: dict with all splits 
    '''
    # take all cols for X if feature_cols is unspecified, otherwise subset df to incl. only feature_cols
    if feature_cols == None: 
        cols_to_drop = ["id", "is_human", "dataset", "sample_params", "model", "temperature", "prompt_number", "unique_id"] +  (["annotations"] if "annotations" in df.columns else []) # drop annotation if present (only present for dailydialog)
        X = df.drop(columns=cols_to_drop)
    else:
        X = df[feature_cols]

    # if model col is present, make explicit categorical for xgboost
    if "model" in X.columns:
        X["model"] = X["model"].astype("category")

    # subset df to a single outcome col for y 
    y = df[[outcome_col]]

    splits = {}

    # create train, test, val splits based on val_test_size, save to splits dict. If val_test_size = 0.15, 15% val and 15% test (and stratify by y to keep class balance as much as possible)
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
        # get dataset name from df
        for key, value in splits.items():
            value.to_csv(save_path / f"{key}.csv")

    return splits

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

def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)

    args = parser.parse_args()

    return args

def clf_pipeline(df, random_state=129, features=None): 
    # init classifier 
    print("[INFO:] Initializing XGClassifier ...")
    clf = XGBClassifier(enable_categorical=True, use_label_encoder=False, random_state=random_state)

    # creating splits 
    if features: 
        print(f"[INFO:] Creating splits with features: {features} using random state {random_state} ...")
    else: 
        print(f"[INFO:] Creating splits with all features using random state {random_state} ...")
    
    splits = create_split(df, random_state=129, val_test_size=0.15, outcome_col="is_human", verbose=False, feature_cols=features)

    # fit classifier
    print("[INFO:] Fitting classifier ...")
    clf = clf_fit(clf, splits["X_train"], splits["y_train"])

    # evaluate classifier on val set
    print("[INFO:] Evaluating classifier ...")
    clf_report = clf_evaluate(clf, X=splits["X_val"], y=splits["y_val"])

    print(clf_report)

    return splits, clf, clf_report

def main():
    args = input_parse()

    # load data, create splits
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"
    temp = 1 # for now 

    print(args.dataset)

    df = load_metrics(human_dir=datapath / "human_metrics", 
                                    ai_dir=datapath / "ai_metrics",
                                    dataset=args.dataset, temp=temp, 
                                    human_completions_only=True
            )
    # filter 
    df = filter_metrics(df, percent_NA=0.9, percent_zero=0.9, verbose=True, log_file=path.parents[0] / "filtered_metrics_classify_log.txt")

    # do on all features
    splits, clf, clf_report = clf_pipeline(df, random_state=129, features=None)

    # do on selected features
    selected_features = ["sentence_length_median", "proportion_unique_tokens", "oov_ratio"]

    splits, clf, clf_report = clf_pipeline(df, random_state=129, features=selected_features)


if __name__ == "__main__":
    main()