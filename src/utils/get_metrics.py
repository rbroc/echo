'''
utils script for extracting metrics
'''
import pandas as pd
import spacy 
import textdescriptives as td
from evaluate import load
import numpy as np


def get_descriptive_metrics(df:pd.DataFrame, text_column:str, spacy_mdl:str="en_core_web_md", batch_size:int=1, n_process:int=1):
    '''
    Extract low level descriptive features doc_length, n_tokens, n_characters and n_sentences 
    '''
    # load nlp, add td to model
    print(f"[INFO:] Loading SpaCY model '{spacy_mdl}'...")
    nlp = spacy.load(spacy_mdl)
    nlp.add_pipe("textdescriptives/descriptive_stats")
    nlp.add_pipe("textdescriptives/quality")

    # pass txt to pipeline
    text = df[text_column]
    print(f"[INFO:] Passing text from column '{text_column}' to pipeline ...")
    docs = nlp.pipe(text, batch_size=batch_size, n_process=n_process)

    print("[INFO:] Extracting metrics to df ...")
    metrics_df = td.extract_df(docs, include_text=False)

    # subset metrics to only include low level features, then concat
    metrics_df = metrics_df[["doc_length", "n_tokens", "n_characters", "n_sentences"]]
    final_df = pd.concat([df, metrics_df], axis=1)

    return final_df

def get_all_metrics(df:pd.DataFrame, text_column:str, spacy_mdl:str="en_core_web_md", batch_size:int=1, n_process:int=1):
    '''
    Extract all metrics using textdescriptives using nlp.pipe(). 

    Same functionality as td.extract_metrics() but allows for multiprocessing. 
    '''
    # load nlp, add td to model
    print(f"[INFO:] Loading SpaCY model '{spacy_mdl}'...")
    nlp = spacy.load(spacy_mdl)
    nlp.add_pipe("textdescriptives/all")

    # pass txt to pipeline
    text = df[text_column]
    print(f"[INFO:] Passing text from column '{text_column}' to pipeline ...")
    docs = nlp.pipe(text, batch_size=batch_size, n_process=n_process)

    print("[INFO:] Extracting metrics to df ...")
    metrics_df = td.extract_df(docs, include_text=False) 

    # sort columns alphabetically, then concat
    metrics_df = metrics_df.reindex(sorted(metrics_df.columns), axis=1) 
    final_df = pd.concat([df, metrics_df], axis=1)

    return final_df

def convert_to_entropy(perplexity: float):
    '''
    Compute entropy from perplexity 
    (since HF's perplexity is "defined as the exponentiated average negative log-likelihood of a sequence, calculated with exponent base `e`."), we just take log(perplexity) to get entropy.
    '''
    return np.log(perplexity)

def compute_perplexity(texts:list, model_id:str = "gpt2", batch_size:int = 1):
    '''
    Compute perplexity 

    This perplexity "is defined as the exponentiated average negative log-likelihood of a sequence, calculated with exponent base `e`."
    source: https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/main/README.md

    Args:
        texts: list of texts
        model_id: model id 
        batch_size: batch size for processing
    '''
    perplexity = load("perplexity", module_type="metric")

    perplexity_scores = perplexity.compute(
                                            predictions=texts, 
                                            model_id=model_id, 
                                            add_start_token=True, # (default to be able to compute perplexity of first token see: https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py)
                                            batch_size=batch_size
                                            )

    return perplexity_scores

def get_information_metrics(df:pd.DataFrame, text_column:str="completions", model_id:str = "gpt2", batch_size:int = 1):
    '''
    Compute information metrics

    Args:
        df: dataframe
        text_column: name of text column
        model_id: model id
        batch_size: batch size for processing
    '''
    # get text
    texts = df[text_column].tolist()

    # compute perplexity
    print(f"[INFO:] Computing perplexity for {text_column} ...")
    results = compute_perplexity(texts, model_id=model_id, batch_size=batch_size)

    # get perplexity only (returns also mean)
    perplexity_scores = results["perplexities"]

    # convert to entropy
    print(f"[INFO:] Computing entropy ...")
    entropy_scores = [convert_to_entropy(score) for score in perplexity_scores]

    # overwrite existing cols
    df["perplexity"] = perplexity_scores
    df["entropy"] = entropy_scores

    # drop "per_word_perplexity" as it no longer makes sense (not connected to the perplexity metric that was computed)
    df.drop(columns=["per_word_perplexity"], inplace=True)
    
    return df