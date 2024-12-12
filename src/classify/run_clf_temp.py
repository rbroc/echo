import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from sklearn.linear_model import LogisticRegression
from src.utils.classify import clf_pipeline
from utils.split_formatter import TextSplitFormatter


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

    # load text data
    text_formatter = TextSplitFormatter(
        splits_dir=path.parents[2] / "datasets_complete" / "text" / f"temp_{temp}",
        dataset=dataset,
        splits_to_load=["train", "val"]
    )

    splits = text_formatter.load_splits()

    test_split = text_formatter.get_split("test")

    vectorised_splits = text_formatter.vectorise(
        X_col="completions",
        split_to_fit_on="train",
        splits_to_transform=["val"],
        max_features=1000,
    )

    train_X, train_y = text_formatter.get_X_y_data(split_name="train", X_col="tfidf", y_col="is_human")


if __name__ == "__main__":
    main()
