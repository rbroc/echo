'''
Experimenting with prompting
'''
# utils 
import pathlib 

# hf models
from transformers import pipeline 

# data wrangling 
import ndjson
import pandas as pd 

# custom
from pipeline import completions_generator


def create_prompt(datapath): 
    dailymail_cnn = {"task_prefix": "summarize the following news article. Do no copy paste from the news article: "}

    # read data
    with open(datapath) as file:
        data = ndjson.load(file)
    
    # make into dataframe
    df = pd.DataFrame(data)

    # create prompt col 
    df["prompt"] = dailymail_cnn["task_prefix"] + df["source"].copy()
    
    return df 

def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[1] / "datasets" / "dailymail_cnn" / "data.ndjson"
    
    # create prompt data
    df = create_prompt(datapath)
    
    # subset data 
    df = df[:1]

    # define model
    model_name = "google/flan-t5-large"

    # intialise pipeline
    model = pipeline(
        model = model_name,
        task = "summarization",
        trust_remote_code = True,
        device_map = "auto"
    )

    # create completion
    min_len, max_tokens = 5, 100
    completions = completions_generator(df, model, "t5", min_len, max_tokens, outfilepath=path.parents[0]/"test.ndjson")

    # merge (temp)
    merged_df = pd.merge(df, completions, on='id')

    print(merged_df["prompt"]) 
    print(merged_df["t5_completions"])

if __name__ == "__main__":
    main()