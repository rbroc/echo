from transformers import pipeline 
import pandas as pd 
import pathlib
import ndjson
from tqdm import tqdm


def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    msrpath = path.parents[1] / "datasets" / "mrpc"
    msrfile = msrpath / "data.ndjson"
    outfile = path.parents[1] / "datasets"

    # load mrpc data
    print("[INFO:] Loading data ...")
    with open(msrfile) as f:
        msrp = ndjson.load(f)
    
    # make into pandas dataframe 
    msrp_data = pd.DataFrame(msrp)

    # subset for testing 
    msrp_data = msrp_data[:10]

    # load mdl, define task  
    print("[INFO]: Loading model ...")
    model = pipeline(
        model = "google/flan-t5-large", 
        task = "summarization"
    )

    min_len = 5
    max_len = 20 

    # empty lst for completions
    completions = []

    # generate text
    for prompt in tqdm(msrp_data["source"], desc="Generating"):
        completion = model(prompt, min_length=min_len, max_length=max_len)
        completions.append(completion[0])

    # make into dataframe
    msrp_data["ai_completions"] = completions
    
    # convert to json
    msrp_data_json = msrp_data.to_json(orient="records", lines=True)

    # save
    with open(outfile / "test_data.ndjson", "w") as file: 
        file.write(msrp_data_json)

    print(msrp_data)

if __name__ == "__main__":
    main()