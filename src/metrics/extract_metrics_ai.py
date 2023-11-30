import pandas as pd
import spacy
import textdescriptives as td
from argparse import ArgumentParser
import pathlib 


def input_parse():
    parser = ArgumentParser()
    parser.add_argument("-d",
                        "--dataset",
                        type=str,
                        required=True)
    parser.add_argument("-m",
                        "--model",
                        type=str, 
                        required=True)
    args = parser.parse_args()

    return args


def process(dataframe, model):
    print("[INFO] Extracting metrics from AI completions...")
    completions = nlp.pipe(dataframe[f"{model}_completions"])
    
    completion_metrics = td.extract_df(completions,
                                        include_text=False)

    completion_metrics['id'] = dataframe["id"]
    print("...done!")
    
    # sort columns because td changes the orders each time script is run
    completion_metrics = completion_metrics.reindex(sorted(completion_metrics.columns), axis=1)

    return completion_metrics

def extract_ai_metrics():
    args = input_parse()

    prompt_numbers = [1, 2, 3, 4, 5, 6]
    path = pathlib.Path(__file__)

    for prompt in prompt_numbers:
        infile = path.parents[2] / "datasets" / "ai_datasets" / args.model / f"{args.dataset}_prompt_{prompt}.ndjson"
        try: 
            data = pd.read_json(infile, lines=True)
            
            # process 
            results = process(data, args.model)
            
            # save
            outpath = path.parents[2] / "results" / "metrics" / "ai_metrics" / args.model 
            outpath.mkdir(parents=True, exist_ok=True)
            results.to_csv(outpath / f"{args.dataset}_prompt_{prompt}_completions.csv")

        except: 
            print(f"KeyError: Dataset does not exist for prompt {prompt}") 
        
if __name__ == "__main__":
    print("[INFO]: Loading model ...")
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("textdescriptives/all")
    extract_ai_metrics()
