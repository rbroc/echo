'''
utils script for extracting metrics
'''
import pandas as pd
import spacy 
import textdescriptives as td

def get_descriptive_metrics(df:pd.DataFrame, text_column:str, spacy_mdl:str="en_core_web_md"):
    '''
    Extract low level descriptive features doc_length, n_tokens, n_characters and n_sentences 
    '''
    textcol = df[text_column]

    metrics = td.extract_metrics(text=textcol, spacy_model=spacy_mdl, metrics=["descriptive_stats", "quality"])
    subset_metrics = metrics[["doc_length", "n_tokens", "n_characters", "n_sentences"]]

    metrics_df = pd.concat([df, subset_metrics], axis=1)
    
    return metrics_df 


def get_all_metrics(df:pd.DataFrame, text_column:str, spacy_mdl:str="en_core_web_md"):
    '''
    Extract all metrics using textdescriptives
    '''
    textcol = df[text_column]

    metrics = td.extract_metrics(text=textcol, spacy_model=spacy_mdl)
    metrics_df = pd.concat([df, metrics], axis=1)
    
    return metrics_df