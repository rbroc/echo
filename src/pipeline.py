from transformers import pipeline, AutoTokenizer
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

def load_file(datafile):
    # load data
    print("[INFO:] Loading data ...")
    with open(datafile) as f:
        data = ndjson.load(f)
    
    # make into dataframe
    df = pd.DataFrame(data)
    
    return df 

def model_picker(chosen_model:str="t5"): 
    '''
    Function for picking model to finetune.

    Args:
        chosen_model: name of model to use. 
    
    Returns:
        full_name: full string name of model 
    '''
    if chosen_model == "falcon":
        full_name = "tiiuae/falcon-7b"

    if chosen_model == "falcon-instruct":
        full_name = "tiiuae/falcon-7b-instruct"

    if chosen_model == "t5": 
        full_name = "google/flan-t5-large"        

    return full_name

def completions_generator(df, model, min_len, max_len, outfile):
    # empty list for completions
    completions = []

    # generate the text
    for prompt in tqdm(df["source"], desc="Generating"):
        completion = model(prompt, min_length=min_len, max_length=max_len, top_k = 10)

        # extraxt ONLY the text from the completion (it is wrapped as a list of dicts otherwise)
        completion_txt = list(completion[0].values())[0]

        # append to lst 
        completions.append(completion_txt)
    
    # make completions into a pandas dataframe    
    df["ai_completions"] = completions

    # convert to json
    df_json = df.to_json(orient="records", lines=True)

    # save it
    with open(outfile, "w") as file:
        file.write(df_json)

    return df


def main(): 
    # intialise arguments 
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[1] / "datasets" / args.filename
    datafile = datapath / "data.ndjson"

    outfile = datapath / "full_data.ndjson"

    # load stuff
    df = load_file(datafile)

    # subset 
    df = df[:1]
    
    # choose model
    full_name = model_picker(args.chosen_model)

    # initialise pipeline 
    print("[INFO]: Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(full_name)
    model = pipeline(
        model = full_name,
        tokenizer = tokenizer,
        task = args.task,
        trust_remote_code = True,
        device_map = "auto"
    )

    # define min and max length 
    min_len, max_len = 5, 200

    # generate text and save it to json
    df_json = completions_generator(df, model, min_len, max_len, outfile)

if __name__ == "__main__":
    main()