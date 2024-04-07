'''
Construct classifiers
'''
import pathlib
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

def main():
    # initialize classifier
    clf = XGBClassifier(enable_categorical=True, use_label_encoder=False)

    # load data, create splits
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "metrics"

    final_df = load_metrics(datapath, dataset="stories", temp=1)

    splits = create_split(final_df, random_state=129, val_test_size=0.15, outcome_col="is_human", verbose=False, feature_cols=["passed_quality_check"])

    # fit classifier
    clf = clf_fit(clf, splits["X_train"], splits["y_train"])

    print(splits["X_train"].columns)

    # evaluate classifier on val set
    clf_report = clf_evaluate(clf, X=splits["X_val"], y=splits["y_val"])

    print(splits["y_test"].value_counts())

    print(clf_report)


if __name__ == "__main__":
    main()