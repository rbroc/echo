'''
Run TFIDF classifier for each dataset and temp combination, save results to /results/classify/clf_results on all features.
'''
import pathlib
import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.classify import clf_pipeline, create_split
from src.utils.process_generations import preprocess_datasets

def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)
    parser.add_argument("-t", "--temp", default=1, help="Temperature of generations", type=float)

    args = parser.parse_args()

    return args

def vectorise(X_train, X_val, X_test, **kwargs):
    # init vectorizer
    vectorizer = TfidfVectorizer(lowercase=False, **kwargs) # lowercase = False as text is already preprocessed

    # fit vectorizer on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # transform validation and test data
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # get feature names
    feature_names = vectorizer.get_feature_names_out()
    print(f"[INFO]: Number of features: {len(feature_names)}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, feature_names

def main(): 
    args = input_parse()
    path = pathlib.Path(__file__)

    # load metrics 
    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    savepath.mkdir(parents=True, exist_ok=True)

    dataset, temp = args.dataset, args.temp

    if temp == 1.0:
        temp = int(temp)

    # loading pca_df (only to ensure that we get the exact same order for the datasets)
    pcapath = path.parents[2] / "results" / "classify" / "pca_results" / "data"
    pca_df = pd.read_csv(pcapath / f"{dataset}_temp{temp}_data.csv")


    pca_df = pca_df[["id", "model", "is_human"]] # subset (we don't want the PC components)
    
    # load completions
    completions_df = preprocess_datasets(
                                        human_dir=path.parents[2] / "datasets" / "human_datasets",
                                        ai_dir=path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA",
                                        models=["beluga7b", "llama2_chat13b", "mistral7b", "llama2_chat7b"],
                                        datasets=[dataset],
                                        prompt_numbers=[21],
                                        temp=temp,
                                        clean_ai=True
                                        )


    # subset to only include 
    completions_df = completions_df[["id", "model", "completions", "dataset", "temperature"]]

    # merge pca_df and completions_df
    df = pca_df.merge(completions_df, on=["id", "model"], how="left")

    # split data 
    splits = create_split(df, feature_cols=["completions"], random_state=129, val_test_size=0.15, outcome_col="is_human", verbose=True)

    # vectorise data
    max_features_tfidf = 1000

    X_train_tfidf, X_val_tfidf, X_test_tfidf, feature_names = vectorise(
                                                                        X_train = splits["X_train"]["completions"].tolist(), 
                                                                        X_val = splits["X_val"]["completions"].tolist(), 
                                                                        X_test = splits["X_test"]["completions"].tolist(),
                                                                        max_features=max_features_tfidf
                                                                        )

    # fit    
    clf, clf_report = clf_pipeline(
                                    df = df, 
                                    model = "LogisticRegression",
                                    X_train = X_train_tfidf,
                                    y_train = splits["y_train"],
                                    X_val = X_val_tfidf,
                                    y_val = splits["y_val"],
                                    feature_names = feature_names,
                                    random_state = 129, 
                                    save_dir = savepath / "clf_reports" / f"{dataset}_temp{temp}", 
                                    save_filename = f"all_models_tfidf_{max_features_tfidf}_features",
                                    )

    

if __name__ == "__main__":
    main()