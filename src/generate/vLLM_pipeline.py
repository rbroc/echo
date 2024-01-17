import argparse
import pathlib
from transformers import set_seed
from vllm import SamplingParams

# custom 
from models import vLLM_Model
from generation import load_json_data, extract_min_max_tokens, login_hf_token, vllm_generate
from prompts import add_prompts_to_df

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-d", "--dataset", help = "pick which dataset you want", type = str, default = "stories")
    parser.add_argument("-mdl", "--model_name", help = "Choose between models ...", type = str, default = "beluga7b")
    parser.add_argument("-prompt_n", "--prompt_number", help = "choose which prompt to use", type = int, default = 1)
    parser.add_argument("-subset", "--data_subset", help = "how many rows you want to include. Useful for testing. Defaults to None.", type = int, default=None)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main(): 
    set_seed(129)

    # init args, define path 
    args = input_parse()
    path = pathlib.Path(__file__)

    ## LOAD DATA ##
    dataset = args.dataset
    datapath = path.parents[2] / "datasets" / "human_datasets" / dataset
    datafile = datapath / "data.ndjson"

    df = load_json_data(datafilepath=datafile, n_subset=args.data_subset)
    _, max_tokens = extract_min_max_tokens(args.dataset)

    # outfilepath 
    if args.data_subset is not None: 
        outpath = path.parents[2] / "datasets" / "ai_vllm_datasets" / f"{args.model_name}" 
    else:
        outpath = path.parents[2] / "datasets" / "ai_vllm_datasets" / "ALL_DATA" / f"{args.model_name}" 

    outpath.mkdir(parents=True, exist_ok=True)

    ## LOAD MDL ##
    print(f"[INFO:] Instantiating model ...")
    chosen_model_name = args.model_name
    cache_models_path =  path.parents[3] / "models"

    # load token (for llama2)
    if "llama2_chat" in chosen_model_name: 
        login_hf_token()

    # load LLM
    model_obj = vLLM_Model(chosen_model_name)

    # format prompts depending on the model 
    prompt_df = add_prompts_to_df(model_obj, df, dataset=dataset, prompt_number=args.prompt_number) 

    ## INIT GEN ##
    print(f"[INFO:] Generating completions with {model_obj.get_model_name()} ...")
    prob_sampling = SamplingParams(temperature=1, top_k=50, top_p=1, presence_penalty=0, frequency_penalty=0, repetition_penalty=1, max_tokens=1055)

    # generate
    df_completions = vllm_generate(
                                                        vllm_model=model_obj, 
                                                        df=prompt_df, 
                                                        prompt_col=f"prompt_{args.prompt_number}", 
                                                        max_tokens=max_tokens, 
                                                        sample_params = prob_sampling, # can be set to NONE to do no sampling
                                                        outfilepath=outpath / f"{dataset}_prompt_{args.prompt_number}.ndjson",
                                                        cache_dir=cache_models_path
                                                        )

    print("[INFO:] DONE!")


if __name__ == "__main__":
    main()