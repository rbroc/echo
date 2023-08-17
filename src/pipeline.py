'''
Pipeline to generate AI completions with various models using Hugging Face's pipeline() function. 
'''

# utils 
import pathlib
import argparse
from tqdm import tqdm

# data wrangling 
import pandas as pd 
import ndjson

# models  
from transformers import pipeline, AutoTokenizer

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

def load_file(filepath):
    '''
    Load ndjson file from path and convert to pandas dataframe 

    Args
        filepath: full path to file 
    
    Returns
        df: pandas dataframe 
    '''

    # load data
    print("[INFO:] Loading data ...")
    with open(filepath) as f:
        data = ndjson.load(f)
    
    # make into dataframe
    df = pd.DataFrame(data)
    
    return df 

def model_picker(chosen_model:str="t5"): 
    '''
    Function for picking model to finetune.

    Args
        chosen_model: name of model to use. 
    
    Returns
        full_name: full string name of model 
        tokenizer: loaded tokenizer if chosen_model = "falcon-7b" or "falcon-instruct" 
    '''
    tokenizer = None

    if chosen_model == "falcon":
        full_name = "tiiuae/falcon-7b"
        tokenizer = AutoTokenizer.from_pretrained(full_name)

    if chosen_model == "falcon-instruct":
        full_name = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(full_name)

    if chosen_model == "t5": 
        full_name = "google/flan-t5-large"        

    return full_name, tokenizer

def completions_generator(df, model, model_name:str, min_len:int , max_tokens: int, outfilepath):
    '''
    Create completions based on source text in dataframe (df). Save to outfilepath.

    Args
        df: dataframe with "source" text col
        model: initalised pipeline
        model_name: name of model (used for naming the column with generated text)
        min_len: minimum length of the completion (output)
        max_tokens: maximum new tokens to be added 
        outfilepath: path where the file should be saved

    Returns
        completions_df: dataframe with model completions and ID 
    '''

    # empty list for completions
    completions = []

    # generate the text
    for prompt in tqdm(df["source"], desc="Generating"):
        completion = model(prompt, min_length=min_len, max_new_tokens=max_tokens)

        # extraxt ONLY the text from the completion (it is wrapped as a list of dicts otherwise)
        completion_txt = list(completion[0].values())[0]

        # append to lst 
        completions.append(completion_txt)
    
    # add ID column from completions_df   
    completions_df = df[["id"]].copy()

    # add completions 
    completions_df[f"{model_name}_completions"] = completions

    # convert to json
    completions_json = completions_df.to_json(orient="records", lines=True)

    # save it
    with open(outfilepath, "w") as file:
        file.write(completions_json)

    return completions_df


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
    df = df[:10]
    
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
    df_json = completions_generator(df, model, args.chosen_model, min_len, max_tokens, outfile)

if __name__ == "__main__":
    main()