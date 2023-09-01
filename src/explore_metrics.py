'''
Script for exploring metrics within the different datasets. Useful for systematising the prompts
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

def get_min_max_doc_length(datapath, filename):
    # define paths
    source_file = datapath / f"{filename}_source.csv"
    completions_file = datapath / f"{filename}_completions.csv"

    # load files
    source = pd.read_csv(source_file)
    completions = pd.read_csv(completions_file)

    # get dictionary
    min_lengths = {
        #f"{filename}_source": round(source["doc_length"].min(), 3),
        f"{filename}_completions": round(completions["doc_length"].min(), 3)
    }
    max_lengths = {
        f"{filename}_completions": round(completions["doc_length"].max(), 3)
    }

    return min_lengths, max_lengths

def get_median_doc_length(datapath, filename):
    # define paths
    source_file = datapath / f"{filename}_source.csv"
    completions_file = datapath / f"{filename}_completions.csv"

    # load files
    source = pd.read_csv(source_file)
    completions = pd.read_csv(completions_file)

    # get dictionary
    median_lengths = {
        f"{filename}_source": round(source["doc_length"].median(), 3),
        f"{filename}_completions": round(completions["doc_length"].median(), 3)
    }

    return median_lengths

#def get_quantiles(datapath, filename):
    # define paths
    #source_file = datapath / f"{filename}_source.csv"
    #completions_file = datapath / f"{filename}_completions.csv"

    # load files
   # source = pd.read_csv(source_file)
    #completions = pd.read_csv(completions_file)

    # get dictionary
    #quantile_lengths = {
        #f"{filename}_source": source["doc_length"].quantile([0.25, 0.50, 0.75]),
       # f"{filename}_completions": completions["doc_length"].quantile([0.25, 0.50, 0.75])
    #}

    #return quantile_lengths

def main(): 
    # init args
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[1] / "out" 

    min_lengths = []
    max_lengths = []
    median_lengths = []
    #quantile_lengths = []

    for filename in ["stories", "dailydialog", "dailymail_cnn", "mrpc"]:
        min_len, max_len = get_min_max_doc_length(datapath, filename)
        median_len = get_median_doc_length(datapath, filename)
        #quantile_len = get_quantiles(datapath, filename)

        min_lengths.append(min_len)
        max_lengths.append(max_len)
        median_lengths.append(median_len)
        #quantile_lengths.append(quantile_len)

    print(f"\n MIN lengths for each dataset \n {min_lengths} \n\n MAX lengths for each dataset \n {max_lengths} \n\n MEDIAN lengths for each dataset \n {median_lengths}")
    #print(f"\n QUANTILES \n {quantile_lengths}")

if __name__ == "__main__":
    main()
