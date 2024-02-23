'''
Inspect data: check lengths of human versus AI 
'''
import pathlib
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import spacy 
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.process_generations import preprocess_datasets

def input_parse():
    parser = argparse.ArgumentParser(description='Inspect data')
    parser.add_argument('--dataset', "-d", type=str, help='dataset to inspect', default="dailydialog")
    args = parser.parse_args()
    return args

def plot_lengths_per_dataset(df, col="doc_length", hue_col='model', bins=30, figsize=(12, 6), title='Doc Lengths by Framework', save_path=None):
    unique_datasets = df['dataset'].unique()
    num_datasets = len(unique_datasets)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    colormap = sns.color_palette("viridis").as_hex()

    for i, dataset in enumerate(unique_datasets):
        # filter dataframe for current dataset
        dataset_df = df[df['dataset'] == dataset]

        # plot each hue (e.g., each model)
        ax = axes[i]
        for j, (hue_val, hue_df) in enumerate(dataset_df.groupby(hue_col, sort=False)):
            if hue_val == "human":
                ax.hist(x=col, bins=bins, data=hue_df, alpha=0.6, label=hue_val, color='red', edgecolor="white")
            else: 
                ax.hist(x=col, bins=bins, data=hue_df, alpha=0.6, label=hue_val, color=colormap[j], edgecolor="white")

        # add contents
        ax.set_title(f'{dataset.upper()}')
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(bottom=0.1)
    fig.supxlabel("Length of Documents (SpaCy doc length)")
    fig.supylabel("Frequency")
    plt.tight_layout(pad=1.5)

    # extract legend from last plot and place it in the in fig.legend underneath suptitle
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=num_datasets, bbox_to_anchor=(0.5, -0.06))

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
    plot_lengths_per_dataset(
                    df, 
                    col="doc_length", 
                    hue_col='model', 
                    bins=30, 
                    figsize=(12, 8), 
                    title = f"Doc Lengths by Framework (Temperature: {temperature})",
                    save_path=path.parents[0] / f"temp{temperature}_doc_lengths.png"
                    )                 

    print(df[["model", "completions", "doc_length"]])
    

if __name__ == "__main__":
    main()