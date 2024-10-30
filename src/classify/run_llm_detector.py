"""
BASELINE: Run a self-made LLM detector on the given dataset
"""
import pandas as pd
import argparse
import pathlib

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

    # get correct format 
    X_train = train_df["completions"].values
    X_val = val_df["completions"].values

    y_train = train_df["is_human"].values
    y_val = val_df["is_human"].values

    labels = [0, 1]

    # zero-shot classification
    X_val = X_val[:2]
    model = Trooper("keeeeenw/MicroLlama")
    model.fit(None, labels)
    y_pred = model.predict(X_val)

    # get rapport
    clf_report = create_eval(y_val, y_pred, val_df, savepath, f"zero_shot_clf_report")

    

if __name__ == "__main__":
    main()