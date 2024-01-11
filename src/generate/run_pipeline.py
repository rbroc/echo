'''
Pipeline to generate AI completions with various models using Hugging Face's pipeline() function. 
'''
import argparse
import pathlib
import ndjson, pandas as pd
from transformers import set_seed

# custom 
from models import FullModel, QuantizedModel
from prompts import add_task_prompt, add_system_prompt

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-d", "--dataset", help = "pick which dataset you want", type = str, default = "stories")
    parser.add_argument("-mdl", "--chosen_model", help = "Choose between ...", type = str, default = "beluga7b")
    parser.add_argument("-prompt_n", "--prompt_number", help = "choose which prompt to use", type = int, default = 1)
    parser.add_argument("-subset", "--data_subset", help = "how many rows you want to include. Useful for testing. Defaults to None.", type = int, default=None)
    parser.add_argument("-batch", "--batch_size", help = "Batching of dataset. Mainly for processing in parallel for GPU. Defaults to no batching (batch size of 1). ", type = int, default=1)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def extract_min_max_tokens(dataset: str):
    '''
    Return a specific min, max tokens for a dataset

    Args
        dataset: name of dataset 
    '''
    valid_datasets = {
        "dailymail_cnn": (6, 433),
        "stories": (112, 1055),
        "mrpc": (8, 47),
        "dailydialog": (2, 220)
    }

    if dataset not in valid_datasets:
        valid_datasets_str = ", ".join(valid_datasets.keys())
        raise ValueError(f"Invalid dataset '{dataset}'. Choose from {valid_datasets_str}")

    return valid_datasets[dataset]

def load_token(model_name:str): 
    if "llama2" in model_name:
        from huggingface_hub import login

        # get token from txt
        with open(path.parents[2] / "tokens" / "hf_token.txt") as f:
            hf_token = f.read()

        login(hf_token)

def main(): 
    # seed, only really necessary if do_sample = True in generation_pipeline (set to False as default)
    set_seed(129)

    # init args, define path 
    args = input_parse()
    path = pathlib.Path(__file__)

    ## LOAD DATA ##
    datapath = path.parents[2] / "datasets" / "human_datasets" / args.dataset
    datafile = datapath / "data.ndjson"

    print("[INFO:] Loading data ...")
    with open(datafile) as f:
        data = ndjson.load(f)
    
    df = pd.DataFrame(data)

    # subset data for prompting. saves to "datasets_ai" / "chosen_model". If data is not subsetted, will save data to full_data / "chosen_model"
    if args.data_subset is not None: 
        df = df[:args.data_subset]
        outpath = path.parents[2] / "datasets" / "ai_datasets" / f"{args.chosen_model}" 

    if args.data_subset is None:
        outpath = path.parents[2] / "datasets" / "ai_datasets" / "ALL_DATA" / f"{args.chosen_model}" 

    outpath.mkdir(parents=True, exist_ok=True)

    # define min and max generation length for dataset (from filename)
    min_len, max_tokens = extract_min_max_tokens(args.dataset)

    ## LOAD MDL ##
    print(f"[INFO:] Instantiating model ...")
    model_name = args.chosen_model
    cache_models_path =  path.parents[3] / "models"

    # load token (for llama2)
    load_token(model_name)
    
    # load full or quantized model depending on model_name
    if "Q" not in model_name: 
        model_instance = FullModel(model_name)
    else: 
        model_instance = QuantizedModel(model_name)

    ## DEF PROMPTS ##
    if model_name in ["beluga", "llama2_chat"]:
        prompt_df = add_system_prompt(df, model_name, args.dataset, args.prompt_number)
    else:
        prompt_df = add_task_prompt(df, args.dataset, args.prompt_number)

    ## INIT GEN ## 
    print(f"[INFO:] Generating completions with {model_instance.get_model_name()} ...")

    prob_sampling = {"do_sample":True, "temperature":1}

    df_completions = model_instance.completions_generator(
                                                          df=prompt_df, 
                                                          prompt_col=f"prompt_{args.prompt_number}", 
                                                          min_len=min_len,
                                                          max_tokens=max_tokens, 
                                                          batch_size=args.batch_size, 
                                                          sample_params = prob_sampling,
                                                          outfilepath=outpath, 
                                                          cache_dir=cache_models_path
                                                          )

if __name__ == "__main__":
    main()