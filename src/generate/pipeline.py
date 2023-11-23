'''
Generation pipeline
'''
from models import FullModel, QuantizedModel
from prompts import add_task_prompt, add_system_prompt
import pandas as pd

def generation_pipeline(chosen_model:str, df:pd.DataFrame, dataset:str, prompt_number:int, min_len:int, max_tokens:int, batch_size:int=1, do_sample=False, outfilepath=None):
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
    # instantiate model
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
    df_completions = model_instance.completions_generator(
                                                          df=prompt_df, 
                                                          prompt_col=f"prompt_{prompt_number}", 
                                                          min_len=min_len,
                                                          max_tokens=max_tokens, 
                                                          batch_size=batch_size, 
                                                          do_sample=do_sample, 
                                                          outfilepath=outfilepath
                                                          )

    return df_completions