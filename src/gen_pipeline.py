'''
Pipeline to generate AI completions with various models using Hugging Face's pipeline() function. 
'''

# utils 
import argparse
import pathlib

# custom functions for datasets
from utils.text_generation.data_fns import load_file, extract_min_max_tokens

# custom function for pipeline 
from utils.text_generation.pipeline_fns import generation_pipeline

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-f", "--filename", help = "pick which dataset you want", type = str, default = "stories")
    parser.add_argument("-mdl", "--chosen_model", help = "Choose between ...", type = str, default = "beluga")
    parser.add_argument("-prompt_n", "--prompt_number", help = "choose which prompt to use", type = int, default = 1)
    parser.add_argument("-subset", "--data_subset", help = "how many rows you want to include. Useful for testing. Defaults to None.", type = int, default=None)
    parser.add_argument("-batch", "--batch_size", help = "Batching of dataset. Mainly for processing in parallel for GPU. Defaults to no batching (batch size of 1). ", type = int, default=1)

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

    outpath = path.parents[1] / "datasets_ai" / f"{args.chosen_model}" 
    loggingpath = path.parents[1] / "datasets_ai" / "logs" / f"{args.chosen_model}" 

    for p in [outpath, loggingpath]:
        p.mkdir(parents=True, exist_ok=True)

    df = load_file(datafile)

    # subset (temporary for testing)
    if args.data_subset is not None: 
        df = df[:args.data_subset]

    # define min and max length 
    min_len, max_tokens = extract_min_max_tokens(args.filename)

    if "llama2" in args.chosen_model:
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
        batch_size=args.batch_size,
        outfilepath = outpath / f"{args.filename}_prompt_{args.prompt_number}.ndjson",
        loggerpath = loggingpath,
        loggername = f"{args.filename}_prompt_{args.prompt_number}"
        )


if __name__ == "__main__":
    main()