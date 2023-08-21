'''
Functions for running the text generation pipelines 
'''

# utils 
from tqdm import tqdm

# data wrangling 
import pandas as pd 
import ndjson

# models  
from transformers import pipeline, AutoTokenizer

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


def completions_generator(df, prompt_col:str, model, model_name:str, min_len:int , max_tokens: int, outfilepath=None):
    '''
    Create completions based on source text in dataframe (df). Save to outfilepath if specified.

    Args
        df: dataframe with "source" text col
        prompt_col: name of column to generate completions from 
        model: initalised pipeline
        model_name: name of model (used for naming the column with generated text)
        min_len: minimum length of the completion (output)
        max_tokens: maximum new tokens to be added 
        outfilepath: path where the file should be saved (defaults to none, not saving anything)

    Returns
        completions_df: dataframe with model completions and ID 
    '''

    # empty list for completions
    completions = []

    # generate the text
    for prompt in tqdm(df[prompt_col], desc="Generating"):
        completion = model(prompt, min_length=min_len, max_new_tokens=max_tokens)

        # extraxt ONLY the text from the completion (it is wrapped as a list of dicts otherwise)
        completion_txt = list(completion[0].values())[0]

        # append to lst 
        completions.append(completion_txt)
    
    # add ID column from completions_df   
    completions_df = df[["id"]].copy()

    # add completions 
    completions_df[f"{model_name}_completions"] = completions

    # save it to json ONLY if outfilepath is specified 
    if outfilepath is not None:
        completions_json = completions_df.to_json(orient="records", lines=True)

        with open(outfilepath, "w") as file:
            file.write(completions_json)

    return completions_df

