'''
Functions for generating data with HF pipeline and vLLM
'''
import ndjson
import pathlib
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from vllm import SamplingParams
import spacy
import random

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
        "dailydialog": (4, 112)
    }

    if dataset not in valid_datasets:
        valid_datasets_str = ", ".join(valid_datasets.keys())
        raise ValueError(f"Invalid dataset '{dataset}'. Choose from {valid_datasets_str}")

    return valid_datasets[dataset]

def login_hf_token(token_path=pathlib.Path(__file__).parents[2] / "tokens" / "hf_token.txt"): 
    '''
    Load HF token from "tokens" folder and login. 
    '''
    from huggingface_hub import login

    # get token from txt
    with open(token_path) as f:
        hf_token = f.read()

    login(hf_token)

def load_json_data(datafilepath:pathlib.Path(), n_subset:int=None):
    '''
    Load ndjson data and optionally subset it. 

    Args
        datafilepath: pathlib path to datafile 
        subset_n: n first rows to subset. Defaults to None (i.e., no subset)

    Returns
        df: pandas dataframe 
    '''
    print("[INFO:] Loading data ...")
    with open(datafilepath) as f:
        data = ndjson.load(f)

    df = pd.DataFrame(data)
    
    if n_subset: 
        print(f"[INFO:] Data subsetted to size n = {n_subset}")
        df = df[:n_subset]

    return df 

def hf_generate(hf_model, df:pd.DataFrame, prompt_col:str="prompt_1", min_len:int=112, max_tokens:int=1055, batch_size=1, sample_params:dict=None, outfilepath=None, cache_dir=None):
    '''
    Generate data with instantiated hf_model using the custom Class FullModel or QuantizedModel (see models.py)

    Args
        hf_model: FullModel or QuantizedModel object 
        df: dataframe with prompt col 
        prompt_col: name of column to generate completions from 
        min_len: minimum length of the completion (output)
        max_tokens: maximum new tokens to be added 
        batch_size: the amount of batches the data should be handled in (default to 1, i.e., no batching).
        sample_params: if specified, will be used to do probabilistic decoding. 
        outfilepath: path where the file should be saved (defaults to none, not saving anything)
        cache_dir: path to load model if saved locally (defaults to None, downloading the model from the hub)

    Returns
        completions_ds: huggingface dataset with model completions and ID 
    '''
    # intialize mdl
    hf_model.initialize_model(cache_dir=cache_dir)

    # convert to HF dataset for batching/streaming option
    ds = Dataset.from_pandas(df)

    completions = []    
    temp_counter = 0
    temp_threshold = batch_size*2 # threshold for amount of completions before it saves a remp file 

    for out in tqdm(hf_model.model(KeyDataset(ds, prompt_col), min_length=min_len, max_new_tokens=max_tokens, batch_size=batch_size, **sample_params)): 
        completion_txt = list(out[0].values())[0] # retrieve only raw text 
        completions.append(completion_txt)
        
        if outfilepath: 
            temp_counter += 1
            if temp_counter % temp_threshold == 0:
                print("[INFO]: Saving temp file...")
                temp_df = df.iloc[:len(completions)].copy()
                temp_df = temp_df.drop(columns=["human_completions", "source"], errors='ignore')
                temp_df[f"{hf_model.chosen_model_name}_completions"] = completions
                temp_df["sample_params"] = str(sample_params)

                temp_df.to_json(outfilepath, orient="records", lines=True, force_ascii=False)

    # add completions + sample params
    df[f"{hf_model.chosen_model_name}_completions"] = completions
    df["sample_params"] = str(sample_params)
    final_df = df.drop(columns=["human_completions", "source"])

    if outfilepath:
        print(f"[INFO]: Saving data to {outfilepath}...")
        final_df.to_json(outfilepath, orient="records", lines=True, force_ascii=False)

    return final_df  

def vllm_generate(vllm_model, df:pd.DataFrame, prompt_col:str="prompt_1", min_tokens:int=None, max_tokens:int=1055, sample_params=None, outfilepath=None, cache_dir=None):
    '''
    Generate data with instantiated vllm model 

    Args
        vllm_model: VLLMMODEL object
        df: dataframe with prompt col 
        prompt_col: name of column to generate completions from 
        min_tokens: minimum length of the completion (output). If specified, will generate new completions for rows that are too short (as vllm has no min_len param)
        max_tokens: maximum new tokens to be added 
        sample_params: SampleParms Object
        outfilepath: path where the file should be saved (defaults to none, not saving anything)
        cache_dir: path to load model if saved locally (defaults to None, downloading the model from the hub)
    '''
    # intialize mdl
    vllm_model.initialize_model(cache_dir=cache_dir)

    completions = []    

    # convert prompt col to list
    prompts = df[prompt_col].tolist()

    # generate outputs 
    sample_params_obj = SamplingParams(**sample_params)
    outputs = vllm_model.model.generate(prompts, sample_params_obj)

    # save outputs
    for output in outputs: 
        completion = output.outputs[0].text
        completions.append(completion)

    # drop cols
    df = df.drop(columns=["human_completions", "source"])

    # add col
    df[f"{vllm_model.chosen_model_name}_completions"] = completions
    df["sample_params"] = str(sample_params)

    if min_tokens:
        try: 
            print("[INFO]: Checking length of completions...")
            nlp = spacy.blank("en")
            df["doc_length"] = df[f"{vllm_model.chosen_model_name}_completions"].apply(lambda x: len(nlp(x)))

            # Initially, check all rows for too short completions
            too_short_ids = df[df["doc_length"] < min_tokens]["id"].tolist()

            sample_params["n"] = 2  # Starting value for 'n'
            while too_short_ids and sample_params["n"] <= 30:
                print(f"[INFO]: Generating new completions for {len(too_short_ids)} too short rows with n = {sample_params['n']}...")
                new_df = df[df["id"].isin(too_short_ids)]
                new_prompts = new_df[prompt_col].tolist()

                sample_params_obj = SamplingParams(**sample_params)
                too_short_outputs = vllm_model.model.generate(new_prompts, sample_params_obj)

                for idx, output in enumerate(too_short_outputs):
                    valid_completions = [comp.text for comp in output.outputs if len(nlp(comp.text)) >= min_tokens]
                    
                    if valid_completions:
                        random_completion = random.choice(valid_completions)
                        completion_id = too_short_ids[idx]
                        df.loc[df["id"] == completion_id, f"{vllm_model.chosen_model_name}_completions"] = random_completion
                        df.loc[df["id"] == completion_id, "doc_length"] = len(nlp(random_completion))

                # Check if there are still too short completions left
                df["doc_length"] = df[f"{vllm_model.chosen_model_name}_completions"].apply(lambda x: len(nlp(x)))
                too_short_ids = df[df["doc_length"] < min_tokens]["id"].tolist()

                sample_params["n"] += 1  # Increment 'n' for the next iteration
            
        except: 
            print("[WARNING]: Error Occured. Likely CUDA problems.")

        if too_short_ids:
            print(f"[WARNING]: len({len(too_short_ids)}) completions still too short after max iterations.")
      
    if outfilepath is not None:
        print(f"[INFO]: Saving data to {outfilepath}...")
        df.to_json(outfilepath, orient="records", lines=True, force_ascii=False)

