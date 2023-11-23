'''
Script for loading model pipelines
'''
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd 
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class Model(): 
    def __init__(self, chosen_model):
        self.chosen_model = chosen_model
        self.model_name = self.get_model_name()
        self.model = None

    @abstractmethod
    def initialize_model(self):
        '''Intialize model using pipeline(). Subclasses use this differently'''
        pass

    def get_model_name(self): 
        '''
        Get full model name from a shorter, chosen name.
        '''

        model_names = {
            "beluga7b":"stabilityai/StableBeluga-7B", 
            "beluga70b":"stabilityai/StableBeluga2",
            "llama2_chat13b":"meta-llama/Llama-2-13b-chat-hf",
            "beluga70bQ":"TheBloke/StableBeluga2-70B-GPTQ", # GPTQ optimized for GPU
            "llama_chat70bQ":"TheBloke/Llama-2-70B-Chat-GPTQ"
        }

        try: 
            return model_names.get(self.chosen_model)
        except KeyError as e: 
            all_names = ", ".join(model_names.keys())
            raise ValueError(
                "Passed model is not supported, "
                f"please choose one of {all_names}"
            ) from e 

    def completions_generator(self, df:pd.DataFrame, prompt_col:str, min_len:int, max_tokens:int, batch_size=1, do_sample=False, outfilepath=None):
        '''
        Create completions based on source text in dataframe (df). Allows for batching inference (NB. GPU needed!).

        Args
            df: dataframe with prompt col 
            prompt_col: name of column to generate completions from 
            min_len: minimum length of the completion (output)
            max_tokens: maximum new tokens to be added 
            batch_size: the amount of batches the data should be handled in (default to 1, i.e., no batching).
            do_sample: whether the model should do greedy decoding (False) or some kind of sampling.
            outfilepath: path where the file should be saved (defaults to none, not saving anything)

        Returns
            completions_ds: huggingface dataset with model completions and ID 
        '''
        # intialize mdl 
        self.initialize_model() 

        # convert to HF dataset for batching/streaming option
        ds = Dataset.from_pandas(df)

        # use pipeline for dataset 
        completions = []
     
        for out in tqdm(self.model(KeyDataset(ds, prompt_col), min_length=min_len, max_new_tokens=max_tokens, batch_size=batch_size, do_sample=do_sample)): 
            completion_txt = list(out[0].values())[0] # retrieve only raw text 
            completions.append(completion_txt)
        
        # remove human cols, add model completions to ds 
        completions_ds = ds.remove_columns(["human_completions", "source"])
        completions_ds = completions_ds.add_column(f"{self.chosen_model}_completions", completions)
        
        if outfilepath is not None:
            completions_ds.to_json(outfilepath, orient="records", lines=True, force_ascii=False)

        return completions_ds 

class FullModel(Model):
    '''
    Full, unquantized models (Beluga and Llama2)
    '''
    def initialize_model(self):
        if self.model is None: 
            self.model = pipeline(
                model=self.get_model_name(),  # get mdl name from base class
                torch_dtype=torch.bfloat16,
                device_map = "auto",
                return_full_text=False
            )
            
            # allow for padding 
            self.model.tokenizer.pad_token_id = self.model.model.config.eos_token_id

class QuantizedModel(Model):
    '''
    Quantized GPQT models e.g., https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ (optimised for GPU).
    '''
    def initialize_model(self):
        if self.model is None: 
            model_name = self.get_model_name()

            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")
                
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
            self.model = pipeline(
                    model=model_name,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map = "auto",
                    return_full_text=False
                )