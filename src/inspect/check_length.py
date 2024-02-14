'''
Inspect data: check lengths of human versus AI 
'''
import pathlib
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import spacy 
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.process_generations import preprocess_datasets

def input_parse():
    parser = argparse.ArgumentParser(description='Inspect data')
    parser.add_argument('--dataset', "-d", type=str, help='dataset to inspect', default="dailydialog")
    args = parser.parse_args()
    return args

def plot_distribution(df, col="doc_length", hue_col='model', bins=30, figsize=(10, 6), title='Doc Lengths by Framework \n (Beluga 7B, Stories Dataset (total: 2000 generations))', save_path=None):
    sns.set(style="whitegrid")

    # plot
    plt.figure(figsize=figsize)
    sns.histplot(data=df, x='doc_length', hue=hue_col, kde=False, bins=bins, palette='viridis')

    # adjust
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
    plt.title(title)
    plt.xlabel('Length of Documents (SpaCy doc length)')
    plt.ylabel('Frequency')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def main(): 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"
    human_dir = path.parents[2] / "datasets" / "human_datasets"

    args = input_parse()

    models = ["beluga7b", "llama2_chat13b", "mistral7b"]

    print("[INFO:] Preprocessing datasets ...")
    df = preprocess_datasets(ai_dir = ai_dir, human_dir = human_dir, models=models, datasets=[args.dataset], temp = "temp1")

    # filtered df 
    filtered_df = df[(df["prompt_number"] == "21") | (df["model"] == "human") & (df["dataset"] == args.dataset)]

    # identify NA doc lengths and fill them in with spacy
    nlp = spacy.blank("en")

    for i, row in filtered_df.iterrows():
        if pd.isna(row["doc_length"]):
            doc = nlp(row["completions"])
            filtered_df.at[i, "doc_length"] = len(doc)

    # plot
    plot_distribution(
                    filtered_df, 
                    col="doc_length", 
                    hue_col='model', 
                    bins=30, 
                    figsize=(10, 6), 
                    title=f'Doc Lengths by Framework \n ({args.dataset.upper()} Dataset (total: {len(filtered_df)} generations))', 
                    save_path=path.parents[0] / f"{args.dataset}_doc_lengths.png"
                    )                 

    print(filtered_df[["model", "completions", "doc_length"]])
    

if __name__ == "__main__":
    main()