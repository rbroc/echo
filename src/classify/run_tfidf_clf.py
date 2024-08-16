'''
Run TFIDF classifier for each dataset and temp combination, save results to /results/classify/clf_results on all features.
'''
import pathlib
import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.classify import clf_pipeline
from src.utils.process_metrics import load_metrics

def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)
    parser.add_argument("-t", "--temp", default=1, help="Temperature of generations", type=float)

    args = parser.parse_args()

    return args

def vectorise(X_train, X_val, X_test):
    # init vectorizer
    vectorizer = TfidfVectorizer(lowercae=False) # lowercase = False as text is already preprocessed
    
    # fit vectorizer on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # transform validation and test data
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_val_tfidf, X_test_tfidf

def main(): 
    args = input_parse()

    # paths
    path = pathlib.Path(__file__)

    # load metrics 
    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    savepath.mkdir(parents=True, exist_ok=True)

    dataset, temp = args.dataset, args.temp

    if temp == 1.0:
        temp = int(temp)

    df = load_metrics(
                        human_dir=datapath / "human_metrics", 
                        ai_dir=datapath / "ai_metrics", 
                        dataset=dataset, 
                        temp=temp, 
                        human_completions_only=True
                        )
    
