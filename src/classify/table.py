'''
Present txt file as table

Code from: https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/src/modules/visualisation.py
'''
import pathlib
import argparse
import pandas as pd

from great_tables import GT

def create_df_from_clf_txt(filepath:pathlib.Path, skiprows=[0, 1, 2], nrows=5):
    '''
    Create a dataframe from a text file containing the classification report from sklearn.metrics.classification_report

    Args:
        filepath: path to text file
        skiprows: rows to skip when reading the file (e.g. if there is text above the table)
        nrows: number of rows to read from the file

    Returns: 
        df: dataframe containing the classification report 
    '''
    # read only (skip first three rows and last three rows)
    df = pd.read_csv(filepath, skiprows=skiprows, nrows=nrows)

    # replace macro avg and weighted avg with macro_avg and weighted_avg
    df.iloc[:,0]= df.iloc[:,0].str.replace(r'(macro|weighted)\savg', r'\1_avg', regex=True)

    # split the columns by whitespace
    df = df.iloc[:,0].str.split(expand=True)

    # define new column names
    new_cols = ['class', 'precision', 'recall', 'f1-score', 'support']
    df.columns = new_cols

    # identify the row with the accuracy score 
    is_accuracy = df['class'] == 'accuracy'

    # move the accuracy row values into the precision and recall columns (they are placed incorrectly when the columns are split)
    df.loc[is_accuracy, ['f1-score', 'support']] = df.loc[is_accuracy, ['precision', 'recall']].values

    # set precision and recall to None for the accuracy row
    df.loc[is_accuracy, ['precision', 'recall']] = None

    # transpose
    df = df.T

    # name columns (where 0 and 1 refers to the classes)
    df.columns = ['0', '1', 'accuracy', 'macro_avg', 'weighted_avg']

    # drop class index
    df = df[~df.index.isin(["class"])]

    return df

def input_parse():
    parser = argparse.ArgumentParser()

    # add dataset as arg 
    parser.add_argument("-d", "--dataset", default="dailymail_cnn", help="Choose between 'stories', 'dailymail_cnn', 'mrpc', 'dailydialog'", type=str)
    parser.add_argument("-t", "--temp", default=1, help="Temperature of generations", type=float)

    args = parser.parse_args()

    return args    


def prepare_df_for_table(df, type:str):
    # avg_df
    avg_df = df[["accuracy", "macro_avg", "weighted_avg"]]

    # class df
    class_df = df[["0", "1"]]

    # unstack
    class_df = class_df.unstack().to_frame().T

    # rename so that columns are called 0_precision, 0_recall, 0_f1-score, 0_support, 1_precision, etc.
    class_df.columns = [f"{col}_{row}" for row, col in class_df.columns]

    # add type col
    class_df["type"] = type

    # sort cols 
    class_df = class_df[["type", "precision_0", "precision_1", "recall_0", "recall_1", "f1-score_0", "f1-score_1", "support_0", "support_1"]]

    # add accuracy
    class_df["accuracy"] = avg_df.loc["f1-score", "accuracy"]

    # rename f1-score to f1_score
    class_df.columns = class_df.columns.str.replace("-", "_")

    return class_df

def create_table(df, savepath:pathlib.Path): 
    table = GT(df)
    table = table.tab_spanner(label="Precision", columns=["precision_0", "precision_1"])
    table = table.tab_spanner(label="Recall", columns=["recall_0", "recall_1"])
    table = table.tab_spanner(label="F1-score", columns=["f1_score_0", "f1_score_1"])
    table = table.tab_spanner(label="Support", columns=["support_0", "support_1"])

    table = table.cols_label(
                            precision_0="0", 
                            precision_1="1", 
                            recall_0="0", 
                            recall_1="1", 
                            f1_score_0="0", 
                            f1_score_1="1", 
                            support_0="0", 
                            support_1="1",
                            accuracy="Accuracy",
                            type="Type"
                            )

    table = table.opt_vertical_padding(scale=2).opt_horizontal_padding(scale=3)

    # save table
    html = table.as_raw_html("table.html")

    # save as html
    with open(savepath / "all_results.html", "w") as f:
        f.write(html)

def main(): 
    args = input_parse()

    path = pathlib.Path(__file__)

    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    datapath = savepath / "clf_reports" / f"{args.dataset}_temp{args.temp}" 

    dfs = []

    # get filepaths 
    filepaths = [file for file in datapath.iterdir() if file.suffix == ".txt"]

    # sort filepaths by name
    filepaths = sorted(filepaths, key=lambda x: x.stem)

    for file in filepaths:
        df = create_df_from_clf_txt(file)
        table_df = prepare_df_for_table(df, file.stem)
        dfs.append(table_df)

    final_df = pd.concat(dfs)

    # create table
    create_table(final_df, datapath)



if __name__ == "__main__":
    main()

