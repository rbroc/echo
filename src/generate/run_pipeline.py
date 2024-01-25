'''
Pipeline to generate AI completions with various models. 

Run the script in the terminal by typing:
    python src/generate/run_pipeline.py -d {DATASET} -mdl {MODEL_NAME} -prompt_n {PROMPT_NUMBER} -subset {DATA_SUBSET} -temperature {TEMPERATURE}

Args
    -d -> dataset, choose between "stories", "dailymail_cnn", "mrpc" and "dailydialog"
    -mdl -> model name, choose between "beluga7b", "beluga70b", "beluga70bQ", "llama2_chat13b", "llama2_chat70bQ", "mistral7b", "mistral8x7b"
    -prompt_n -> task prompt defined in prompts.py for each dataset. Choose between 1 to 6.
    -subset -> how many rows to include in the generations. Defaults to None (i.e., all rows). Otherwise takes the n first rows. Mainly for testing. 
    -temperature -> decoding param (controls stochasticity of the generations)

The additional flag -hf will trigger an Hugging Face implmentation of the model as opposed to vLLM: 
    python src/generate/run_pipeline.py -hf -batch {BATCH_SIZE}

NOTE: the -batch argument which allows processing in parallel with HF (vLLM does this automatically). 
'''
import argparse
import pathlib
from transformers import set_seed

# custom 
from models import FullModel, QuantizedModel, vLLM_Model
from generation import load_json_data, extract_min_max_tokens, login_hf_token, hf_generate, vllm_generate
from prompts import add_prompts_to_df

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-d", "--dataset", help = "pick which dataset you want", type = str, default = "stories")
    parser.add_argument("-mdl", "--model_name", help = "Choose between models ...", type = str, default = "beluga7b")
    parser.add_argument("-prompt_n", "--prompt_number", help = "choose which prompt to use", type = int, default = 1)
    parser.add_argument("-subset", "--data_subset", help = "how many rows you want to include. Useful for testing. Defaults to None.", type = int, default=None)
    parser.add_argument("-temperature", "--temperature", help = "temperature for decoding. Defaults to 1.", type = float, default=1)
    parser.add_argument("-batch", "--batch_size", help = "Batching of dataset. Only for HF for processing in parallel for GPU. Defaults to no batching (batch size of 1). ", type = int, default=1)
    parser.add_argument("-hf", "--use_hf_pipeline", help="Use HF pipeline if set, otherwise use vLLM", action='store_true')

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main():
    # only relevant for probability sampling / decoding 
    set_seed(129)

    # args, pathing 
    args = input_parse()
    path = pathlib.Path(__file__)

    # load data
    dataset  = args.dataset
    datapath = path.parents[2] / "datasets" / "human_datasets" / dataset
    datafile = datapath / "data.ndjson"

    df = load_json_data(datafilepath=datafile, n_subset=args.data_subset)
    min_len, max_tokens = extract_min_max_tokens(dataset)

    # chose model 
    chosen_model_name = args.model_name
    cache_models_path =  path.parents[3] / "models"

    # load token (for llama2)
    if "llama2_chat" in chosen_model_name: 
        login_hf_token()

    # define sampling params (temp defaults to 1)
    params = {"temperature":args.temperature, "top_k":50, "top_p":1, "repetition_penalty":1, "length_penalty":1} 

    # run pipeline
    if args.use_hf_pipeline:
        hf_pipeline(args, df, min_len, max_tokens, path, chosen_model_name, cache_models_path, sample_params=params)
    else: 
        vllm_pipeline(args, df, max_tokens, path, chosen_model_name, cache_models_path, sample_params=params)
    

def hf_pipeline(args, df, min_len, max_tokens, path, chosen_model_name, cache_models_path=None, sample_params:dict=None):    
    '''
    Generation steps specific to a model implementation in HF

    Args
        args: CLI arguments
        df: dataframe with data to generate completions for
        min_len: minimum number of tokens to generate
        max_tokens: maximum number of tokens to generate
        path: path to this script
        cache_models_path: path to cache models (if None, will redownload models)
        chosen_model_name: name of the model to use
        sample_params: dictionary of sampling params to use

    Returns
        df_completions: dataframe with completions
    '''
    if args.data_subset is None:
        outpath = path.parents[2] / "datasets" / "ai_datasets" / "HF" / "FULL_DATA" / chosen_model_name
    else:
        outpath = path.parents[2] / "datasets" / "ai_datasets" / "HF" / "SUBSET_DATA" / chosen_model_name

    outpath.mkdir(parents=True, exist_ok=True)

    # init model object -> full or quantized model depending on mdl name (mdl will first be loaded in completions_generator). 
    print(f"[INFO:] Instantiating model ...")
    if "Q" not in chosen_model_name: 
        model_obj = FullModel(chosen_model_name)
    else: 
        model_obj = QuantizedModel(chosen_model_name)

    # format prompts depending on model # 
    prompt_df = add_prompts_to_df(model_obj, df, dataset=args.dataset, prompt_number=args.prompt_number) 

    # set prob_sampling 
    all_params=None

    if sample_params: 
        hf_params = {"do_sample":True}
        all_params = {**hf_params, **sample_params}
        print(f"Decoding params: {all_params}")

    temperature = int(args.temperature) if args.temperature % 1 == 0 else args.temperature 

    print(f"[INFO:] Generating completions with {model_obj.full_model_name} ...")
    df_completions = hf_generate(
        hf_model=model_obj,
        df=prompt_df, 
        prompt_col=f"prompt_{args.prompt_number}", 
        min_len=min_len,
        max_tokens=max_tokens, 
        batch_size=args.batch_size, 
        sample_params = all_params,
        outfilepath=outpath / f"{args.dataset}_prompt_{args.prompt_number}_temp{temperature}.ndjson",
        cache_dir=cache_models_path
    )
    print("[INFO:] HF Pipeline DONE!")

    return df_completions

def vllm_pipeline(args, df, max_tokens, path, chosen_model_name, cache_models_path=None, sample_params:dict=None):
    '''
    Generation steps specific to a model implementation with vLLM (https://github.com/vllm-project/vllm)

    Args
        args: CLI arguments
        df: dataframe with data to generate completions for
        max_tokens: maximum number of tokens to generate
        path: path to this script
        cache_models_path: path to cache models (if None, will redownload models)
        chosen_model_name: name of the model to use
        sample_params: dictionary of sampling params to use. 

    Returns
        df_completions: dataframe with completions
    '''
    if args.data_subset is None:
        outpath = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA" / chosen_model_name
    else:
        outpath = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "SUBSET_DATA" / chosen_model_name

    outpath.mkdir(parents=True, exist_ok=True)

    # load LLM
    model_obj = vLLM_Model(chosen_model_name)

    # format prompts depending on the model 
    prompt_df = add_prompts_to_df(model_obj, df, dataset=args.dataset, prompt_number=args.prompt_number) 

    # setup 
    all_params=None

    if sample_params: 
        vllm_params = {"presence_penalty":0, "frequency_penalty":0, "max_tokens": max_tokens} # penalties set to 0 as they do not exist in HF framework. Max tokens is defined in sample params here unlike HF.
        all_params = {**vllm_params, **sample_params}
        print(f"Decoding params: {all_params}")

    temperature = int(args.temperature) if args.temperature % 1 == 0 else args.temperature 

    print(f"[INFO:] Generating completions with {model_obj.full_model_name} ...")
    df_completions = vllm_generate(
        vllm_model=model_obj, 
        df=prompt_df, 
        prompt_col=f"prompt_{args.prompt_number}", 
        max_tokens=max_tokens, 
        sample_params = all_params,
        outfilepath=outpath / f"{args.dataset}_prompt_{args.prompt_number}_temp{temperature}.ndjson",
        cache_dir=cache_models_path
    )
    print("[INFO:] vLLM Pipeline DONE!")

    return df_completions

if __name__ == "__main__":
    main()