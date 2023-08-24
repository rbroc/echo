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

# import prompting 
from modules.prompt_fns import PromptGenerator, BelugaPromptGenerator

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
        model_name: full string name of model 
    '''
    if chosen_model == "falcon":
        model_name = "tiiuae/falcon-7b"

    if chosen_model == "falcon_instruct":
        model_name = "tiiuae/falcon-7b-instruct"

    if chosen_model == "t5": 
        model_name = "google/flan-t5-xxl"        

    if chosen_model == "beluga": 
        model_name = "stabilityai/StableBeluga-7B"    
    
    if chosen_model == "llama2":
        model_name = "meta-llama/Llama-2-7b-hf"

    return model_name

def completions_generator(df, prompt_col:str, model, model_name:str, min_len:int , max_tokens: int, outfilepath=None):
    '''
    Create completions based on source text in dataframe (df). Save to outfilepath if specified.

    Args
        df: dataframe with "source" text col
        prompt_col: name of column to generate completions from 
        model: initalised model pipeline
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
    completions_df = df[["id", prompt_col]].copy()

    # add completions 
    completions_df[f"{model_name}_completions"] = completions

    # save it to json ONLY if outfilepath is specified 
    if outfilepath is not None:
        completions_json = completions_df.to_json(orient="records", lines=True, force_ascii=False)

        with open(outfilepath, "w", encoding = "utf-8") as file:
            file.write(completions_json)

    return completions_df

def generation_pipeline(chosen_model, df, datafile, prompt_number, min_len, max_tokens, outfilepath=None):
    # create prompts (specific to the model)
    if chosen_model == "beluga":
        pg = BelugaPromptGenerator(prompt_number)
        df = pg.create_beluga_prompt(df, datafile)
    else: 
        pg = PromptGenerator(prompt_number)
        df = pg.create_prompt(df, datafile)

    # retrive full name of chosen model 
    model_name = model_picker(chosen_model)
    
    # initialise model (different initialisation for falcon & llama2)
    print("[INFO]: Loading model ...")
    if "falcon" in chosen_model:
        # intialise tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = pipeline(
            model = model_name,
            tokenizer = tokenizer, 
            trust_remote_code = True,
            device_map = "auto",
            return_full_text=False
        )

    elif "llama2" in chosen_model: 
        model = pipeline(
            model = model_name,
            device_map = "auto",
            return_full_text=False
        )
    
    else: 
        model = pipeline(
            model = model_name,
            device_map = "auto"
        )

    # generate text and save it to json 
    df_completions = completions_generator(df, f"prompt_{prompt_number}", model, chosen_model, min_len, max_tokens, outfilepath=outfilepath) 

    return df