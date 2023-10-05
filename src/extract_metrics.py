import pandas as pd
import spacy
import textdescriptives as td
from argparse import ArgumentParser


def path_loaders():
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True)
    args = parser.parse_args()

    return args


def process(dataframe):
    print("[INFO] Extracting metrics from sources...")
    # process source text
    ids = dataframe['id']
    sources = nlp.pipe(dataframe["source"])
    # extract the metrics as a dataframe
    source_metrics = td.extract_df(sources,
                                   include_text=False)
    source_metrics['id'] = ids
    print("... done!")
    # process completions
    print("[INFO] Extracting metrics from completions...")
    completions = nlp.pipe(dataframe["human_completions"])
    completion_metrics = td.extract_df(completions,
                                        include_text=False)
    completion_metrics['id'] = ids

    # sort cols 
    source_metrics = source_metrics.reindex(sorted(source_metrics.columns), axis=1)
    completion_metrics = completion_metrics.reindex(sorted(completion_metrics.columns), axis=1)

    print("...done!")

    return source_metrics, completion_metrics


def main():
    args = path_loaders()
    infile = "datasets/" + args.input + '/data.ndjson'
    # change to required file
    data = pd.read_json(infile, lines=True)
    # process with spacy
    source_results, completion_results = process(data)
    # get filepaths
    source_outfile = "out/" + args.input.split()[0] + "_source.csv"
    completion_outfile = "out/" + args.input.split()[0] + "_completions.csv"
    # save
    source_results.to_csv(source_outfile)
    completion_results.to_csv(completion_outfile)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("textdescriptives/all")
    main()
