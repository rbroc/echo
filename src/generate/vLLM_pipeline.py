import pathlib
from transformers import set_seed
from vllm import LLM, SamplingParams
from prompts import add_system_prompt
import ndjson, pandas as pd

def main(): 
    set_seed(129)

    # init args, define path 
    path = pathlib.Path(__file__)

    ## LOAD DATA ##
    dataset = "stories"
    prompt_number = 1
    datapath = path.parents[2] / "datasets" / "human_datasets" / "stories"
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
    model_name = "stabilityai/StableBeluga2"
    model = LLM(model_name, download_dir=cache_models_path)
    prob_sampling = SamplingParams(temperature=0.8)

    # generate
    completions = []

    outputs = model.generate(df[f"prompt_{prompt_number}"], prob_sampling)

    for output in outputs: 
        completion = output.outputs[0].text
        completions.append(completion)

    # add to df
    df["completions"] = completions

    # select only completions and prompt number
    df = df[["id", f"prompt_{prompt_number}", "completions"]]

    # save df to json
    df.to_json("test.json")


if __name__ == "__main__":
    main()