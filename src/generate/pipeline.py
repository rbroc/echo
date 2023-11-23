'''
Generation pipeline
'''
from models import FullModel, QuantizedModel
from prompt_fns import SpecialPromptGenerator, PromptGenerator
import pandas as pd

def generation_pipeline(chosen_model:str, df:pd.DataFrame, datafile:str, prompt_number:int, min_len:int, max_tokens:int, batch_size:int=1, do_sample=False, outfilepath=None):
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

    Returns
        df_completions: dataframe with completions
    '''
    if "Q" not in chosen_model: 
        model_instance = FullModel(chosen_model)
    else: 
        model_instance = QuantizedModel(chosen_model)

    # intialise prompt generator
    pg = SpecialPromptGenerator(prompt_number, chosen_model) if chosen_model in ["beluga", "llama2_chat"] else PromptGenerator(prompt_number)

    # create prompt 
    df = pg.create_prompt(df, datafile)

    # create completions with completions generator from BaseModel
    df_completions = model_instance.completions_generator(df=df, prompt_col=f"prompt_{prompt_number}", min_len=min_len, max_tokens=max_tokens, batch_size=batch_size, do_sample=do_sample, outfilepath=outfilepath)

    return df_completions