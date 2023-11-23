'''
Helper functions for datasets. Contains functions for loading the datasets, and extracting min/max new tokens. 
'''
from tqdm import tqdm
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

def extract_min_max_tokens(datafilename):
    if "stories" in datafilename:
        min_tokens = 112
        max_new_tokens = 1055
    
    if "mrpc" in datafilename:
        min_tokens = 8
        max_new_tokens = 47
    
    if "dailymail_cnn" in datafilename:
        min_tokens = 6
        max_new_tokens = 433
    
    if "dailydialog" in datafilename: 
        min_tokens = 2
        max_new_tokens = 220

    return min_tokens, max_new_tokens