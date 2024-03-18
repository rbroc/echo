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
from src.generate.generation import extract_min_max_tokens

def plot_distribution_per_dataset(df, min_max_tokens_dict=None, col="doc_length", bins=30, figsize=(12, 6), title='Doc Lengths by Framework', save_path=None, caption=False):
    unique_datasets = df['dataset'].unique()
    num_datasets = len(unique_datasets)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    colormap = sns.color_palette("viridis").as_hex()

    # custom xlims, ylims for each dataset
    custom_x_lims = {"stories": (0, 1100), "mrpc": (0, 50), "dailydialog": (0, 250), "dailymail_cnn": (0, 450)}
    custom_y_lims = {"stories": (0, 1000), "mrpc": (0, 800), "dailydialog": (0, 2000), "dailymail_cnn": (0, 850)}

    for i, dataset in enumerate(unique_datasets):
        # filter dataframe for current dataset
        dataset_df = df[df['dataset'] == dataset]

        # plot each hue (e.g., each model, but not human)
        ax = axes[i]
        for j, (hue_val, hue_df) in enumerate(dataset_df.groupby("model", sort=False)):
            if hue_val != "human":
                ax.hist(x=col, bins=bins, data=hue_df, alpha=0.8, label=hue_val, color=colormap[j], edgecolor="white")

        #  plot human
        human_df = dataset_df[dataset_df["model"] == "human"]
        if not human_df.empty:
            ax.hist(x=col, bins=bins, data=human_df, alpha=0.5, label="human", color='red', edgecolor="white")

        # add contents
        ax.set_title(f'{dataset.upper()}')
        ax.set_xlabel('')
        ax.set_ylabel('')

        # custom xlims, ylims
        if dataset in custom_x_lims:
            ax.set_xlim(custom_x_lims[dataset])
        if dataset in custom_y_lims:
            ax.set_ylim(custom_y_lims[dataset])

        # add min, max tokens line
        if min_max_tokens_dict and dataset in min_max_tokens_dict:
            min_tokens, max_tokens = min_max_tokens_dict[dataset]
            ax.axvline(x=min_tokens, color='black', linestyle='--')
            ax.axvline(x=max_tokens, color='black', linestyle='--')

    if min_max_tokens_dict and caption: 
        fig.text(x=0.5, y=-0.09, s=f"Dashed lines represent the pre-defined min and max tokens for each dataset {min_max_tokens_dict}", ha='center', fontsize=8) 

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
    # paths 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[2] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"
    human_dir = path.parents[2] / "datasets" / "human_datasets"
    save_dir = path.parents[2] / "results" / "lengths"
    save_dir.mkdir(parents=True, exist_ok=True)

    # specs (models, datasets, temperatures)
    models = ["beluga7b", "llama2_chat13b", "mistral7b"]
    datasets = ["dailydialog", "dailymail_cnn", "mrpc", "stories"]
    temperatures = [1, 1.5, 2]

    # extract min, max tokens
    min_max_tokens_dict = {}

    for dataset in datasets:
        min_max_tokens_dict[dataset] = extract_min_max_tokens(dataset)

    for temp in temperatures:
        print(f"[INFO:] Preprocessing datasets for temperature: {temp} ...")
        df = preprocess_datasets(ai_dir = ai_dir, human_dir = human_dir, models=models, datasets=datasets, temp = temp, prompt_numbers=[21, 22])

        # identify NA doc lengths and fill them in with spacy
        nlp = spacy.blank("en")

        print("[INFO:] Extracting doc lengths for NA rows...")
        for i, row in df.iterrows():
            if pd.isna(row["doc_length"]):
                doc = nlp(row["completions"])
                df.at[i, "doc_length"] = len(doc)

        # get min max for each dataset only for model = human
        human_df = df[df["model"] == "human"]
        
        for dataset in datasets:
            dataset_df = human_df[human_df["dataset"] == dataset]
            min_doc_length = dataset_df["doc_length"].min()
            max_doc_length = dataset_df["doc_length"].max()
            print(f"[INFO:] Dataset: {dataset}, min doc length: {min_doc_length}, max doc length: {max_doc_length}")

        # plot
        print("[INFO:] Plotting data ...")
        plot_distribution_per_dataset(
                        df, 
                        min_max_tokens_dict,
                        col="doc_length", 
                        bins=30, 
                        figsize=(12, 8), 
                        title = f"Doc Lengths (Temperature: {temp})",
                        save_path=save_dir / f"temp{temp}_doc_lengths.png", 
                        caption=True
                        )                 

        print(df[["model", "completions", "doc_length"]])

    print("[INFO:] DONE!")

if __name__ == "__main__":
    main()