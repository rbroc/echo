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

def plot_distribution(df, col="doc_length", hue_col='model', bins=30, figsize=(10, 6), title='Doc Lengths by Framework', save_path=None):
    sns.set(style="whitegrid")

    unique_datasets = df['dataset'].unique()
    num_datasets = len(unique_datasets)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for i, dataset in enumerate(unique_datasets):
        # filter dataframe for current dataset
        dataset_df = df[df['dataset'] == dataset]

        # plot
        sns.histplot(data=dataset_df, x=col, hue=hue_col, kde=False, bins=bins, palette='viridis', ax=axes[i])

        # adjust
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(100))
        axes[i].set_title(f'{dataset}')
    
    fig.text(0.5, 0.04, 'Length of Documents (SpaCy doc length)', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')

    fig.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def main(): 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"
    human_dir = path.parents[2] / "datasets" / "human_datasets"

    args = input_parse()

    models = ["beluga7b", "llama2_chat13b", "mistral7b"]
    datasets = ["dailydialog", "dailymail_cnn", "mrpc", "stories"]
    temperature = 1

    print("[INFO:] Preprocessing datasets ...")
    df = preprocess_datasets(ai_dir = ai_dir, human_dir = human_dir, models=models, datasets=datasets, temp = temperature, prompt_numbers=[21, 22])

    # identify NA doc lengths and fill them in with spacy
    nlp = spacy.blank("en")

    for i, row in df.iterrows():
        if pd.isna(row["doc_length"]):
            doc = nlp(row["completions"])
            df.at[i, "doc_length"] = len(doc)

    # plot
    plot_distribution(
                    df, 
                    col="doc_length", 
                    hue_col='model', 
                    bins=50, 
                    figsize=(10, 6), 
                    save_path=path.parents[0] / f"temp{temperature}_doc_lengths.png"
                    )                 

    print(df[["model", "completions", "doc_length"]])
    

if __name__ == "__main__":
    main()