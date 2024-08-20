'''
Present txt file as table

Code from: https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/src/modules/visualisation.py
'''
import pathlib
import pandas as pd

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

    return df


def create_multi_header(df:pd.DataFrame, filename:str="test.html", savepath:pathlib.Path=None):
    '''
    hacky solution (for now)
    '''
    df_class = df[df['class'].isin(['0', '1'])]

    classes = [0, 1]
    metrics = ['precision', 'recall', 'f1-score', 'support']

    # create a multi-index dataframe
    df_multi = pd.DataFrame(index=classes, columns=pd.MultiIndex.from_product([metrics, classes]))

    # fill in the multi-index dataframe
    for c in classes:
        for m in metrics:
            df_multi[m, c] = df_class.loc[str(c), m]

    # print first row
    table_df = df_multi.iloc[0].to_frame().T

    # add col
    table_df["type"] = "all_models_all_features"

    # place type first in columns
    table_df = table_df[["type", "precision", "recall", "f1-score", "support"]]

    # drop index
    table_df.reset_index(drop=True, inplace=True)

    # save 
    table_df.to_html("test.html")


def main(): 
    path = pathlib.Path(__file__)

    savepath = path.parents[2] / "results" / "classify" / "clf_results"
    filepath = savepath / "clf_reports" / "dailymail_cnn_temp1" / "all_models_all_features.txt"

    df = create_df_from_clf_txt(filepath)

    df_class = df[df['class'].isin(['0', '1'])]
    df_non_class = df[2:]
    
    # Set the 'class' column as the index
    df_class.set_index('class', inplace=True)

    # drop class column
    df_class = df_class[['precision', 'recall', 'f1-score', 'support']]

    classes = [0, 1]
    metrics = ['precision', 'recall', 'f1-score', 'support']

    # create a multi-index dataframe
    df_multi = pd.DataFrame(index=classes, columns=pd.MultiIndex.from_product([metrics, classes]))

    # fill in the multi-index dataframe
    for c in classes:
        for m in metrics:
            df_multi[m, c] = df_class.loc[str(c), m]

    # print first row
    table_df = df_multi.iloc[0].to_frame().T

    # add col
    table_df["type"] = "all_models_all_features"

    # place type first in columns
    table_df = table_df[["type", "precision", "recall", "f1-score", "support"]]

    # drop index
    table_df.reset_index(drop=True, inplace=True)

    # save 
    table_df.to_html("test.html")




if __name__ == "__main__":
    main()

