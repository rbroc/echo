'''
Pipeline to generate AI completions with various models using Hugging Face's pipeline() function. 
'''

# utils 
import pathlib
import argparse

# import custom functions 
from modules.pipeline_fns import load_file, generation_pipeline

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-f", "--filename", help = "pick which dataset you want", type = str, default = "stories")
    parser.add_argument("-mdl", "--chosen_model", help = "Choose between ...", type = str, default = "t5")
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
    min_len, max_tokens = 30, 50

    if args.chosen_model == "llama2":
        from huggingface_hub import login

        # get token from txt
        with open(path.parents[1] / "tokens" / "hf_token.txt") as f:
            hf_token = f.read()

        login(hf_token)

    # run pipeline 
    completions_df = generation_pipeline(
        chosen_model = args.chosen_model, 
        df = df, 
        datafile = args.filename, 
        prompt_number = args.prompt_number, 
        min_len = min_len, 
        max_tokens = max_tokens, 
        outfilepath = path.parents[0]/"test.ndjson")


if __name__ == "__main__":
    main()