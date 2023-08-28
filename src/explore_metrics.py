'''
Script for exploring metrics within the different datasets. Useful for systematising the prompts for T5
'''

# utils
import pathlib 
import argparse

# data wrangling
import pandas as pd

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-f", "--filename", help = "pick which dataset you want", type = str, default = "stories")

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def get_mean_doc_length(datapath, filename):
    # define paths
    source_file = datapath / f"{filename}_source.csv"
    completions_file = datapath / f"{filename}_completions.csv"

    # load files
    source = pd.read_csv(source_file)
    completions = pd.read_csv(completions_file)

    # get dictionary
    mean_lengths = {
        f"{filename}_source": round(source["doc_length"].mean(), 3),
        f"{filename}_completions": round(completions["doc_length"].mean(), 3)
    }

    return mean_lengths

def get_median_doc_length(datapath, filename):
    # define paths
    source_file = datapath / f"{filename}_source.csv"
    completions_file = datapath / f"{filename}_completions.csv"

    # load files
    source = pd.read_csv(source_file)
    completions = pd.read_csv(completions_file)

    # get dictionary
    mean_lengths = {
        f"{filename}_source": round(source["doc_length"].median(), 3),
        f"{filename}_completions": round(completions["doc_length"].median(), 3)
    }

    return mean_lengths


def main(): 
    # init args
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[1] / "out" 

    mean_lengths = []
    median_lengths = []

    for filename in ["stories", "dailydialog", "dailymail_cnn", "mrpc"]:
        mean_len = get_mean_doc_length(datapath, filename)
        median_len = get_median_doc_length(datapath, filename)

        mean_lengths.append(mean_len)
        median_lengths.append(median_len)

    print(f"\n MEAN lengths for each dataset \n {mean_lengths} \n\n MEDIAN lengths for each dataset \n {median_lengths}")

if __name__ == "__main__":
    main()
