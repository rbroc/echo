'''
Classes and functions for generating text for the pipeline.
'''

# utils
from tqdm import tqdm

# data wrangling (for completions)
import pandas as pd 
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

# models  
from transformers import pipeline, AutoTokenizer
import torch

# import prompting 
from utils.text_generation.prompt_fns import PromptGenerator, SpecialPromptGenerator


class BaseModel():
    '''
    Base model for generating text.
    '''

    def __init__(self, chosen_model):
        self.chosen_model = chosen_model
        self.model_name = self.get_model_name() # store model name as an instance var
        self.model = None # intialize as None. Later represented as HF pipeline. 
        
    def get_model_name(self): 
        '''
        Get full model name from a shorter, chosen name.
        '''

        model_names = {
            "falcon": "tiiuae/falcon-7b",
            "falcon_instruct": "tiiuae/falcon-7b-instruct", 
            "t5":"google/flan-t5-xxl",
            "beluga":"stabilityai/StableBeluga-7B", 
            "llama2": "meta-llama/Llama-2-7b-hf", 
            "llama2_chat":"meta-llama/Llama-2-13b-chat-hf"
        }

        return model_names.get(self.chosen_model)

    def initialize_model(self): 
        '''
        Initialize model if not already done. Common method for pipeline. Overridden by subclasses
        '''
        if self.model is None: 
            self.model = pipeline(model = self.model_name, device_map = "auto")


    def completions_generator(self, df:pd.DataFrame, prompt_col:str, min_len:int, max_tokens:int, loggerpath, loggername:str, batch_size=1, outfilepath=None):
        '''
        Create completions based on source text in dataframe (df). Save to outfilepath if specified.

        Args
            df: dataframe with prompt col 
            prompt_col: name of column to generate completions from 
            min_len: minimum length of the completion (output)
            max_tokens: maximum new tokens to be added 
            batch_size: the amount of batches the data should be handled in (default to 1, i.e., no batching).
            outfilepath: path where the file should be saved (defaults to none, not saving anything)

        Returns
            completions_ds: huggingface dataset with model completions and ID 
        '''
        # intialise logger
       # logger = custom_logging("generator", loggername, loggerpath)

        # intialize mdl 
        self.initialize_model() 

        # convert to HF dataset for batching/streaming option
        ds = Dataset.from_pandas(df)

        # empty list for completions
        completions = []

        # use pipeline on dataset
        for out in tqdm(self.model(KeyDataset(ds, prompt_col), min_length=min_len, max_new_tokens=max_tokens, batch_size=batch_size)): 
            completion_txt = list(out[0].values())[0] # retrieve only the raw text 
            #logger.info(completion_txt)
            completions.append(completion_txt)

        # make completions ds without human completions and source
        completions_ds = ds.remove_columns(["human_completions", "source"])

        # add completions to ds  
        completions_ds = completions_ds.add_column(f"{self.chosen_model}_completions", completions)
        
        if outfilepath is not None:
            completions_ds.to_json(outfilepath, orient="records", lines=True, force_ascii=False)

        return completions_ds 

class BelugaModel(BaseModel):
    def __init__(self):
        super().__init__(chosen_model="beluga") # only one chosen_model, so it is specified here, inherited by BaseModel. 

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

class Llama2Model(BaseModel):
    def initialize_model(self):
        if self.model is None: 
            self.model = pipeline(
                model=self.get_model_name(),  # get mdl name from base class
                torch_dtype=torch.bfloat16,
                device_map="auto",
                return_full_text=False
            )

            # allow for padding 
            self.model.tokenizer.pad_token_id = self.model.model.config.eos_token_id

class FalconModel(BaseModel):
    def initialize_model(self):
        if self.model is None:
            # init tokenizer for falcon 
            tokenizer = AutoTokenizer.from_pretrained(self.get_model_name(), padding_side="left")  # get mdl name from base class

            self.model = pipeline(
                model=self.get_model_name(),  
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True, # trust remote code for falcon
                device_map="auto",
                return_full_text=False, 
            )

            # allow for padding 
            self.model.tokenizer.pad_token_id = self.model.model.config.eos_token_id

def generation_pipeline(chosen_model:str, df:pd.DataFrame, datafile:str, prompt_number:int, min_len:int, max_tokens:int, loggerpath, loggername:str, batch_size:int=1, outfilepath=None):
    '''
    Generation pipeline. Create prompts and completions from "source" column. 

    Args
        chosen_model: model_name (e.g., beluga, falcon, falcon_instruct, t5, llama2, llama2chat)
        df: pandas dataframe with "source" column
        datafile: name of datafile
        prompt_number: int (from 1-6)
        min_len: minimum length of generation
        max_tokens: max new tokens to be generate
        outfilepath: path where the datafile with completions should be saved. Defaults to None

    Returns
        df_completions: dataframe with completions
    '''

    if chosen_model == "beluga":
        model_instance = BelugaModel()
    
    elif "llama2" in chosen_model:
        model_instance = Llama2Model(chosen_model)

    elif "falcon" in chosen_model: 
        model_instance = FalconModel(chosen_model)

    else:
        model_instance = BaseModel(chosen_model)  # init BaseModel for other models than the specified ones

    # intialise prompt generator
    pg = SpecialPromptGenerator(prompt_number, chosen_model) if chosen_model in ["beluga", "llama2_chat"] else PromptGenerator(prompt_number)

    # create prompt 
    df = pg.create_prompt(df, datafile)

    # create completions with completions generator from BaseModel
    df_completions = model_instance.completions_generator(df=df, prompt_col=f"prompt_{prompt_number}", min_len=min_len, max_tokens=max_tokens, batch_size=batch_size, outfilepath=outfilepath, loggerpath=loggerpath, loggername=loggername)

    return df_completions