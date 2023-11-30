'''
Generation pipeline
'''
from models import FullModel, QuantizedModel
from prompts import add_task_prompt, add_system_prompt
import pandas as pd
import ndjson

def load_file(filepath):
    '''
    Load ndjson file from path and convert to pandas dataframe 

    Args
        filepath: full path to file 
    
    Returns
        df: pandas dataframe 
    '''
    print("[INFO:] Loading data ...")
    with open(filepath) as f:
        data = ndjson.load(f)
    
    df = pd.DataFrame(data)
    
    return df 

def extract_min_max_tokens(dataset: str):
    '''
    Return a specific min, max tokens for a dataset

    Args
        dataset: name of dataset 
    '''
    valid_datasets = {
        "dailymail_cnn": (6, 433),
        "stories": (112, 1055),
        "mrpc": (8, 47),
        "dailydialog": (2, 220)
    }

    if dataset not in valid_datasets:
        valid_datasets_str = ", ".join(valid_datasets.keys())
        raise ValueError(f"Invalid dataset '{dataset}'. Choose from {valid_datasets_str}")

    return valid_datasets[dataset]

def generation_pipeline(chosen_model:str, df:pd.DataFrame, dataset:str, prompt_number:int, min_len:int, max_tokens:int, batch_size:int=1, do_sample=False, outfilepath=None, cache_dir=None):
    '''
    Generation pipeline. Create prompts and completions from "source" column. 

    Args
        chosen_model: model_name (e.g., beluga, falcon, falcon_instruct, t5, llama2, llama2chat)
        df: pandas dataframe with "source" column
        datafile: name of datafile
        prompt_number: int (from 1-6)
        min_len: minimum length of generation
        max_tokens: max new tokens to be generate
        do_sample: whether the model should do greedy decoding (False) or some kind of sampling.
        outfilepath: path where the datafile with completions should be saved. Defaults to None
        cache_dir: path where model is saved locally (defaults to None, downloading the model from the hub)

    Returns
        df_completions: dataframe with completions
    '''
    # instantiate model
    print(f"[INFO:] Instantiating model ...")
    if "Q" not in chosen_model: 
        model_instance = FullModel(chosen_model)
    else: 
        model_instance = QuantizedModel(chosen_model)

    # generate prompts
    if chosen_model in ["beluga", "llama2_chat"]:
        prompt_df = add_system_prompt(df, chosen_model, dataset, prompt_number)
    else:
        prompt_df = add_task_prompt(df, dataset, prompt_number)

    # generate completions
    print(f"[INFO:] Generating completions with {model_instance.get_model_name()} ...")
    df_completions = model_instance.completions_generator(
                                                          df=prompt_df, 
                                                          prompt_col=f"prompt_{prompt_number}", 
                                                          min_len=min_len,
                                                          max_tokens=max_tokens, 
                                                          batch_size=batch_size, 
                                                          do_sample=do_sample, 
                                                          outfilepath=outfilepath, 
                                                          cache_dir=cache_dir
                                                          )

    return df_completions