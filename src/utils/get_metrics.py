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
    # get text
    text = df[text_column]

    # extract metrics, select only relevant cols 
    metrics_df = td.extract_metrics(text=text, spacy_model=spacy_mdl, metrics=["descriptive_stats", "quality"])
    subset_metrics_df = metrics[["doc_length", "n_tokens", "n_characters", "n_sentences"]]

    # combine with df 
    final_df = pd.concat([df, subset_metrics_df], axis=1)
    
    return final_df 


def get_all_metrics(df:pd.DataFrame, text_column:str, spacy_mdl:str="en_core_web_md"):
    '''
    Extract all metrics using textdescriptives
    '''
    # get text
    text = df[text_column]

    # extract text
    metrics_df = td.extract_metrics(text=text, spacy_model=spacy_mdl)

    # combine with df 
    final_df = pd.concat([df, metrics_df], axis=1)
    
    return final_df


def get_all_metrics_pipe(df:pd.DataFrame, text_column:str, batch_size:int=1, n_process:int=1, spacy_mdl:str="en_core_web_md"):
    '''
    Extract all metrics using textdescriptives using nlp.pipe(). 

    Same functionality as td.extract_metrics() but allows for multiprocessing. 
    '''
    # load nlp, add td to model
    nlp = spacy.load(spacy_mdl)
    nlp.add_pipe("textdescriptives/all")

    # get txt
    text = df[text_column]

    # pass txt to pipeline
    docs = nlp.pipe(text, batch_size=batch_size, n_process=n_process)

    # get metrics as df 
    metrics_df = td.extract_df(docs, include_text=False)

    # sort columns alphabetically
    metrics_df = metrics_df.reindex(sorted(metrics_df.columns), axis=1)

    # concat 
    final_df = pd.concat([df, metrics_df], axis=1)

    return final_df