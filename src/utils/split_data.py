'''
Split data into train, test, and validation sets
'''
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

def create_split(df, random_state:int=129, val_test_size:float=0.15, stratify_cols:list=["is_human", "dataset", "model"]):
    '''
    split data into train, test, and validation sets

    Args: 
        df: dataframe to split
        random_state: seed for split for reproducibility
        val_test_size: size of validation and test sets. 0.15 results in 15% val and 15% test. 

    Returns: 
        splits: dict with all splits 
    '''    
    # If val_test_size = 0.15, 15% val and 15% test (stratify by y to somewhat keep class balance)
    print(f"[INFO:] Creating splits using random state {random_state} ...")   
    train_df, test_val_df = train_test_split(df, test_size=val_test_size*2, random_state=random_state, stratify=df[stratify_cols])
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=random_state, stratify=test_val_df[stratify_cols])

    return train_df, val_df, test_df