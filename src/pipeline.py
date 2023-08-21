'''
Pipeline to generate AI completions with various models using Hugging Face's pipeline() function. 
'''

# utils 
import pathlib
import argparse

# models  
from transformers import pipeline

# import custom pipeline fns
from modules.pipeline_fns import load_file, model_picker, completions_generator

# initialize the parser
def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-f", "--filename", help = "pick which dataset you want", type = str, default = "mrpc")
    parser.add_argument("-mdl", "--chosen_model", help = "Choose between ...", type = str, default = "t5")
    parser.add_argument("-tsk", "--task", help = "which task do you want it to do", type = str, default = "summarization")

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

    outfile = datapath / f"{args.chosen_model}_data.ndjson"

    # load stuff
    df = load_file(datafile)

    # subset (temporary for testing)
    df = df[:1]
    
    # choose model (tokenizer is none if not falcon is chosen)
    full_name, tokenizer = model_picker(args.chosen_model)

    # initialise pipeline 
    print("[INFO]: Loading model ...")
    model = pipeline(
        model = full_name,
        tokenizer = tokenizer,
        task = args.task,
        trust_remote_code = True,
        device_map = "auto"
    )

    # define min and max length 
    min_len, max_tokens = 5, 40

    # generate text and save it to json
    df = completions_generator(df, "source", model, args.chosen_model, min_len, max_tokens, outfile)

if __name__ == "__main__":
    main()