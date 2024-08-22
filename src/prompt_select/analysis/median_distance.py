'''
Get median distance for each prompt, each model 
'''

import pathlib

import pandas as pd
from tabulate import tabulate

def compute_medians(df):
    '''
    Compute median distance for each prompt and each model
    '''
    # compute median
    grouped_median = df.groupby(["dataset", "model", "prompt_number"]).median(numeric_only=True).reset_index()
    
    # create pivot table with medians 
    medians_pivot_tbl = grouped_median.pivot(index=["dataset", "model"], columns="prompt_number", values="distance")

    return medians_pivot_tbl

def get_n_lowest_medians(pivot_table, n_vals: int = 2, table_format="github"):
    '''
    Create a table with the n_lowest medians for each model and each prompt, removing repeated dataset and model names.
    '''
    results = []

    last_dataset = None  # Initialize variables to track the last dataset and model

    for index, row in pivot_table.iterrows():
        lowest_medians = row.nsmallest(n_vals)

        dataset, model = index

        if dataset == last_dataset:
            dataset = ""
        else:
            last_dataset = dataset

        first_row = True
        for prompt_number, median_value in lowest_medians.items():
            if first_row:
                first_row = False
                results.append([dataset, model, prompt_number, f"{median_value:.3f}"])
            else:
                results.append(["", "", prompt_number, f"{median_value:.3f}"])

    table = tabulate(results, headers=["dataset", "model", "prompt", "median"], tablefmt=table_format)

    return table

def create_medians_table(pivot_table, table_format="github"):
    '''
    create table with an overview of all models and datasets
    '''
    # make into df
    medians_df = pivot_table.copy().reset_index()
    
    # round vals
    medians_df = medians_df.round(3)
    
    # rm repeated dataset names + "nan" for pretty table 
    medians_df['dataset'] = [dataset if i % 2 != 0 else '' for i, dataset in enumerate(medians_df['dataset'])]
    medians_df.fillna('', inplace=True)

    table = tabulate(medians_df, headers="keys", tablefmt=table_format, showindex=False, floatfmt=".2f")

    return table

def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[3] / "results" / "prompt_select" / "distance"

    df = pd.read_csv(datapath / "distances_all_PC_cols.csv")

    medians = compute_medians(df)

    # create tables 
    n_lowest = get_n_lowest_medians(medians)
    all_table = create_medians_table(medians)

    print(n_lowest)
    print("\n")
    print(all_table)

if __name__ == "__main__":
    main()