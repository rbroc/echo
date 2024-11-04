"""
BASELINE: Run a self-made LLM detector on the given dataset
"""
import pandas as pd
import argparse
from datetime import datetime
import pathlib
from random import sample, seed

from stormtrooper import Trooper
from sklearn import metrics

def create_eval(y_val, y_pred, df, save_dir=None, save_filename: str = "clf_report"): 
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
            file.write(f"{clf_report}\n")
            file.write(
                f"Model(s) compared with human:{[model for model in df['model'].unique() if model != 'human']}\n"
            )

    return clf_report

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

def sample_train(train_df, sample_size:int=5, random_state:int=129):
    """
    Sample the training data to a smaller size for few-shot learning (stratified)
    """
    # split up into the two classes
    class_0 = train_df[train_df["is_human"] == 0]
    class_1 = train_df[train_df["is_human"] == 1]

    # sample from each class
    sample_size_class_0 = sample_size // 2
    sample_size_class_1 = sample_size - sample_size_class_0

    sample_class_0 = class_0.sample(n=sample_size_class_0, random_state=random_state)
    sample_class_1 = class_1.sample(n=sample_size_class_1, random_state=random_state)

    sample_df = pd.concat([sample_class_0, sample_class_1]).sample(frac=1, random_state=random_state)

    X_train_sample = sample_df["completions"].values
    y_train_sample = sample_df["is_human"].values

    return X_train_sample, y_train_sample

def main(): 
    args = input_parse()
    path = pathlib.Path(__file__)

    dataset, temp = args.dataset, args.temp

    if temp == 1.0:
        temp = int(temp)

    datapath = path.parents[2] / "datasets_complete" / "text" / f"temp_{temp}"

    # filter to only include one dataset
    train_df = pd.read_parquet(datapath / f"train_text.parquet")
    val_df = pd.read_parquet(datapath / f"val_text.parquet")

    train_df = train_df[train_df["dataset"] == dataset]
    val_df = val_df[val_df["dataset"] == dataset]

    # get correct format 
    X_val = val_df["completions"].values
    y_val = val_df["is_human"].values

    # sample for few-shot learning (stratified)
    X_train_sample, y_train_sample = sample_train(train_df, sample_size=2, random_state=129)

    labels = [0, 1]

    # zero-shot classification
    model_name = "google/flan-t5-small"
    model = Trooper(model_name)
    model.fit(None, labels)
    y_pred = model.predict(X_val)

    savepath = path.parents[2] / "results" / "classify" / "clf_results" / "clf_reports" / f"{dataset}_temp{temp}"
    savepath.mkdir(parents=True, exist_ok=True)
    clf_report = create_eval(y_val, y_pred, val_df, savepath, f"zero_shot_{model_name.split('/')[1]}")

    # few-shot classification
    model = Trooper(model_name)
    model.fit(X_train_sample, y_train_sample)
    y_pred = model.predict(X_val)

    savepath = path.parents[2] / "results" / "classify" / "clf_results" / "clf_reports" / f"{dataset}_temp{temp}"
    savepath.mkdir(parents=True, exist_ok=True)
    clf_report = create_eval(y_val, y_pred, val_df, savepath, f"few_shot_{model_name.split('/')[1]}")

if __name__ == "__main__":
    main()