'''
Testing script for one-shot T5
'''

# utils
import pathlib
import argparse
from tqdm import tqdm
import pandas as pd 

# custom functions for datasets, prompt generation and model intialisation
from modules.data_fns import load_file, extract_min_max_tokens
from modules.prompt_fns import OneShotGenerator
from modules.pipeline_fns import BaseModel


def oneshot_generation_pipeline(df:pd.DataFrame, datafile:str, prompt_number:int, min_len:int, max_tokens:int, outfilepath=None):
    '''
    One-shot Generation pipeline for T5. Create prompts and completions from "source" column. 

    Args
        df: pandas dataframe with "source" column
        datafile: name of datafile
        prompt_number: int (from 1-6)
        min_len: minimum length of generation
        max_tokens: max new tokens to be generate
        outfilepath: path where the datafile with completions should be saved. Defaults to None

    Returns
        df_completions: dataframe with completions
    '''

    model_instance = BaseModel("t5")  # init BaseModel for other models than the specified ones

    # intialise prompt generator
    pg = OneShotGenerator(prompt_number)

    # create prompt 
    df = pg.create_prompt(df, datafile)

    # create completions with completions generator from BaseModel
    df_completions = model_instance.completions_generator(df, f"prompt_{prompt_number}", min_len, max_tokens, outfilepath=outfilepath)

    return df_completions

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-f", "--filename", help = "pick which dataset you want", type = str, default = "stories")
    parser.add_argument("-prompt_n", "--prompt_number", help = "choose which prompt to use", type = int, default = 1)
    parser.add_argument("-subset", "--data_subset", help = "how many rows you want to include. Useful for testing. Defaults to None.", type = int, default=None)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main(): 
    # intialise arguments 
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[1] / "datasets" / args.filename
    datafile = datapath / "data.ndjson"
    #outfile = datapath / f"{args.chosen_model}_data.ndjson"

    # load stuff
    df = load_file(datafile)

    # subset (temporary for testing)
    if args.data_subset is not None: 
        df = df[:args.data_subset]

    # define min and max length 
    min_len, max_tokens = extract_min_max_tokens(args.filename)

    # run pipeline 
    completions_df = oneshot_generation_pipeline(
        df = df, 
        datafile = args.filename, 
        prompt_number = args.prompt_number, 
        min_len = min_len, 
        max_tokens = max_tokens, 
        outfilepath = path.parents[0]/"test.ndjson")


if __name__ == "__main__":
    main()