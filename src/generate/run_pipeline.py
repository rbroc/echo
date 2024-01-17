'''
Pipeline to generate AI completions with various models using Hugging Face's pipeline() function. 
'''
import argparse
import pathlib
import ndjson, pandas as pd
from transformers import set_seed

# custom 
from models import FullModel, QuantizedModel
from generation import load_json_data, extract_min_max_tokens, login_hf_token, hf_generate
from prompts import add_prompts_to_df

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-d", "--dataset", help = "pick which dataset you want", type = str, default = "stories")
    parser.add_argument("-mdl", "--model_name", help = "Choose between models ...", type = str, default = "beluga7b")
    parser.add_argument("-prompt_n", "--prompt_number", help = "choose which prompt to use", type = int, default = 1)
    parser.add_argument("-subset", "--data_subset", help = "how many rows you want to include. Useful for testing. Defaults to None.", type = int, default=None)
    parser.add_argument("-batch", "--batch_size", help = "Batching of dataset. Mainly for processing in parallel for GPU. Defaults to no batching (batch size of 1). ", type = int, default=1)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main(): 
    # seed, only necessary if prob_sampling params such as temperature is defined
    set_seed(129)

    # init args, define path 
    args = input_parse()
    path = pathlib.Path(__file__)

    ## LOAD DATA ##
    dataset  = args.dataset
    datapath = path.parents[2] / "datasets" / "human_datasets" / dataset
    datafile = datapath / "data.ndjson"

    df = load_json_data(datafilepath=datafile, n_subset=args.data_subset)
    min_len, max_tokens = extract_min_max_tokens(dataset)

    # subset data for prompting. saves to "datasets_ai" / "model_name". If data is not subsetted, will save data to full_data / "model_name"
    if args.data_subset is not None: 
        outpath = path.parents[2] / "datasets" / "ai_datasets" / f"{args.model_name}" 
    else:
        outpath = path.parents[2] / "datasets" / "ai_datasets" / "ALL_DATA" / f"{args.model_name}" 

    outpath.mkdir(parents=True, exist_ok=True)

    ## LOAD MDL ##
    print(f"[INFO:] Instantiating model ...")
    chosen_model_name = args.model_name
    cache_models_path =  path.parents[3] / "models"

    # load token (for llama2)
    if "llama2_chat" in chosen_model_name: 
        login_hf_token()
    
    # init model object -> full or quantized model depending on mdl name (mdl will first be loaded in completions_generator). 
    if "Q" not in chosen_model_name: 
        model_obj = FullModel(chosen_model_name)
    else: 
        model_obj = QuantizedModel(chosen_model_name)

    # format prompts depending on model # 
    prompt_df = add_prompts_to_df(model_obj, df, dataset=dataset, prompt_number=args.prompt_number) 

    ## INIT GEN ## 
    print(f"[INFO:] Generating completions with {model_obj.full_model_name} ...")

    # generate
    prob_sampling = {"do_sample":True, "temperature":1}

    df_completions = hf_generate(
                                                        hf_model=model_obj,
                                                        df=prompt_df, 
                                                        prompt_col=f"prompt_{args.prompt_number}", 
                                                        min_len=min_len,
                                                        max_tokens=max_tokens, 
                                                        batch_size=args.batch_size, 
                                                        sample_params = prob_sampling, # can be set to NONE to do no sampling
                                                        outfilepath=outpath / f"{dataset}_prompt_{args.prompt_number}.ndjson",
                                                        cache_dir=cache_models_path
                                                        )

    print("[INFO:] DONE!")

if __name__ == "__main__":
    main()