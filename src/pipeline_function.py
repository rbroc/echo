from transformers import pipeline 
import pandas as pd 
import pathlib
import ndjson
from tqdm import tqdm
import argparse

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-f", "--filename", help = "pick which dataset you want", type = str, default = "mrpc")
    parser.add_argument("-mdl", "--chosen_model", help = "Choose between ...", type = str, default = "t5")
    parser.add_argument("-tsk", "--task", help = "which task do you want it to do", type = str, default = "summarization")

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def load_files(filename):
    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[1] / "datasets" / filename
    datafile = datapath / "data.ndjson"

    outfile = datapath / "full_data.ndjson"

    # load data
    print("[INFO:] Loading data ...")
    with open(datafile) as f:
        data = ndjson.load(f)
    
    # make into dataframe a
    df = pd.DataFrame(data)
    df = df[:10]
    
    return df, outfile 

def model_picker(chosen_model:str="t5"): 
    '''
    Function for picking model to finetune.

    Args:
        chosen_model: name of model to use. 
    
    Returns:
        model_dict: dictionary with the name of finetune (key) and name of model to be finetuned (value).
    '''
    if chosen_model == "falcon":
        full_name = "tiiuae/falcon-7b"

    if chosen_model == "falcon-instruct":
        full_name = "tiiuae/falcon-7b-instruct"

    if chosen_model == "t5": 
        full_name = "google/flan-t5-large"        

    return full_name

def load_mdl(full_name, task):
    # load mdl, define task  
    print("[INFO]: Loading model ...")
    model = pipeline(
        model = full_name,
        task = task
    )
    min_len = 5 
    max_len = 20

    return model, min_len, max_len

def completions_generator(df, model, min_len, max_len, outfile):
    #empty list for completions
    completions = []

    #generate the text
    for prompt in tqdm(df["source"], desc="Generating"):
        completion = model(prompt, min_length=min_len, max_length=max_len)
        completions.append(completion[0])
    
    #make it into a pandas dataframe
    df["ai_completions"] = completions

    #convert to json
    df_json = df.to_json(orient="records", lines=True)

    #save it
    with open(outfile / "test_data.ndjson", "w") as file:
        file.write(df_json)
    print(df)

    return df_json


def main(): 
    # intialise arguments 
    args = input_parse()
    #load stuff
    df, outfile = load_files(args.filename)
    #choose model
    full_name = model_picker(args.chosen_model)
    #load in the model
    model, min_len, max_len = load_mdl(full_name, args.task)
    #generate text and save it to json
    df_json = completions_generator(df, model, min_len, max_len, outfile)

if __name__ == "__main__":
    main()