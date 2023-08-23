'''
Pipeline to generate AI completions with various models using Hugging Face's pipeline() function. 
'''

# utils 
import pathlib
import argparse

# models  
from transformers import AutoTokenizer, pipeline

# import custom pipeline fns
from modules.pipeline_fns import load_file, completions_generator

from modules.prompt_fns import PromptGenerator, BelugaPromptGenerator

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-f", "--filename", help = "pick which dataset you want", type = str, default = "mrpc")
    parser.add_argument("-mdl", "--chosen_model", help = "Choose between ...", type = str, default = "t5")
    parser.add_argument("-prompt_n", "--prompt_number", help = "choose which prompt to use", type = int, default = 1)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def model_picker(chosen_model:str="t5"): 
    '''
    Function for picking model to finetune.

    Args
        chosen_model: name of model to use. 
    
    Returns
        full_name: full string name of model 
        tokenizer: loaded tokenizer if chosen_model = "falcon-7b" or "falcon_instruct" 
    '''
    tokenizer = None

    if chosen_model == "falcon":
        full_name = "tiiuae/falcon-7b"
        tokenizer = AutoTokenizer.from_pretrained(full_name)

    if chosen_model == "falcon_instruct":
        full_name = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(full_name)

    if chosen_model == "t5": 
        full_name = "google/flan-t5-xl"        

    if chosen_model == "beluga": 
        full_name = "stabilityai/StableBeluga-7B"    

    return full_name, tokenizer

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
    df = df[:5]

    # create prompt
    if args.chosen_model == "beluga":
        beluga_pg = BelugaPromptGenerator(args.prompt_number)
        df = beluga_pg.create_beluga_prompt(df, args.filename)

    else: 
        pg = PromptGenerator(prompt_number=2)
        df = pg.create_prompt(df, args.filename)
    
    # choose model (tokenizer is none if not falcon is chosen)
    full_name, tokenizer = model_picker(args.chosen_model)

    # initialise pipeline 
    print("[INFO]: Loading model ...")
    model = pipeline(
        model = full_name,
        tokenizer = tokenizer,
        # task = args.task,
        trust_remote_code = True,
        device_map = "auto",
        return_full_text=False,
    )

    # define min and max length 
    min_len, max_tokens = 30, 50

    # generate text and save it to json
    df = completions_generator(df, f"prompt_{args.prompt_number}", model, args.chosen_model, min_len, max_tokens, outfilepath=path.parents[0]/"test.ndjson")

if __name__ == "__main__":
    main()