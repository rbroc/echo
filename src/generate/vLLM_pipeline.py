import pathlib
from transformers import set_seed
from vllm import LLM, SamplingParams
from prompts import add_system_prompt
import ndjson, pandas as pd
import torch

def main(): 
    set_seed(129)

    # init args, define path 
    path = pathlib.Path(__file__)

    # get num gpus
    available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

    ## LOAD DATA ##
    dataset = "mrpc"
    prompt_number = 1
    datapath = path.parents[2] / "datasets" / "human_datasets" / dataset
    datafile = datapath / "data.ndjson"

    print("[INFO:] Loading data ...")
    with open(datafile) as f:
        data = ndjson.load(f)
    
    df = pd.DataFrame(data)

    # add system prompt 
    prompt_df = add_system_prompt(df, "beluga", dataset, prompt_number)

    ## LOAD MDL ##
    print(f"[INFO:] Instantiating model ...")
    cache_models_path =  path.parents[3] / "models"

    # load LLM
    #model_name = "stabilityai/StableBeluga2"
    model_name = "stabilityai/StableBeluga-7B"
    model = LLM(model_name, download_dir=cache_models_path, tensor_parallel_size=available_gpus, seed=129)
    prob_sampling = SamplingParams(temperature=1, top_k=50, top_p=1, presence_penalty=0, frequency_penalty=0, repetition_penalty=1, max_tokens=1055)

    # generate
    completions = []

    df = df[:200]

    # convert prompts list 
    prompts = df[f"prompt_{prompt_number}"].tolist()

    outputs = model.generate(prompts, prob_sampling)

    for output in outputs: 
        completion = output.outputs[0].text
        completions.append(completion)

    # add to df
    df["completions"] = completions

    # select only completions and prompt number
    df = df[["id", f"prompt_{prompt_number}", "completions"]]

    print(df["completions"])

    # save df to json
    df.to_json("test.json", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()