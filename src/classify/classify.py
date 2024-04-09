'''
Construct classifiers
'''
import pathlib
import argparse
from xgboost import XGBClassifier
from sklearn import metrics
from prepare_data import load_metrics, create_split

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

def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)

    args = parser.parse_args()

    return args

def main():
    args = input_parse()

    # initialize classifier
    clf = XGBClassifier(enable_categorical=True, use_label_encoder=False, random_state=129)

    # load data, create splits
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"

    df = load_metrics(datapath, dataset=args.dataset, temp=1)

    # xgboost bugs with the dataset with these two cols "Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`", so temporarily dropped 
    # may be because there are some very large values in the bottom (could be a solution to scale instead)
    if args.dataset == "dailymail_cnn":
        df = df.drop(columns=["per_word_perplexity", "perplexity"])

    # make train, val, test splits
    splits = create_split(df, random_state=129, val_test_size=0.15, outcome_col="is_human", verbose=False)

    # fit classifier
    clf = clf_fit(clf, splits["X_train"], splits["y_train"])

    # evaluate classifier on val set
    clf_report = clf_evaluate(clf, X=splits["X_val"], y=splits["y_val"])

    print(splits["y_train"].value_counts())
    print(splits["y_val"].value_counts())
    print(splits["y_test"].value_counts())

    print(clf_report)


if __name__ == "__main__":
    main()