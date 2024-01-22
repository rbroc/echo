import pathlib
import pandas as pd 
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.pca import get_descriptive_metrics, run_PCA
from src.utils.distance import compute_distances

def load_data(model="beluga7b", file="stories_prompt_1.ndjson", vllm=False, root_data_path=pathlib.Path(__file__).parents[2] / "datasets"):
    if vllm:
        datapath = root_data_path / "ai_vllm_datasets" / "ALL_DATA"
    else: 
        datapath = root_data_path / "ai_datasets" / "prob_decoding"

    # load data 
    filepath = datapath / model / file 
    df = pd.read_json(filepath, lines=True)

    return df 

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

    model = "beluga7b"

    # plot 
    vllm = load_data(vllm=True)
    hf = load_data(vllm=False)

    vllm = vllm[:1000]
    hf = hf[:1000]

    # add framework 
    vllm["model"] = "vllm"
    hf["model"] = "hf"

    # drop cols for both
    for df in [vllm, hf]:
        df.drop(columns=["prompt_1"], inplace=True)

    # concatenate 
    df = pd.concat([vllm, hf], axis=0, ignore_index=True)

    ## PCA ## 
    spacy.util.fix_random_seed(129)

    # rename framework col to model
    df.rename(columns={"beluga7b_completions": "completions"}, inplace=True)
    df["dataset"] = "stories"
    df["prompt_number"] = 1

    print("[INFO:] EXTRACTING LOW LEVEL METRICS")
    metrics_df = get_descriptive_metrics(df, "completions", "id")

    print("[INFO:] RUNNING PCA ...")
    pca, final_df = run_PCA(metrics_df, feature_names=["doc_length", "n_tokens", "n_characters", "n_sentences"], n_components=4)

    print(final_df)
    print(pca.explained_variance_ratio_)

    # print mean doc length per framework
    print(final_df.groupby("model")["doc_length"].mean())

    # plot 
    plot_distribution(final_df, hue_col="model", save_path=path.parents[0] / "doc_length_distribution.png")

    # remove rows with doc length below 112 (min length) 
    # since vllm does not have a min tokens param, it is to avoid those where it is bugged to make a more fair comparison with hf framework implementation
    final_df = final_df[final_df["doc_length"] > 112]

    # make sure that there is 2 of each id (one for each model). If not, drop the row
    final_df = final_df.groupby("id").filter(lambda x: len(x) == 2)

    print(len(final_df))

    print("[INFO:] COMPUTING & PLOTTING DISTANCES ...")
    distance_df = compute_distances(final_df, models=["vllm"], baseline="hf")
    sns.set(style="whitegrid")
    g = sns.catplot(data=distance_df, x="prompt_number", y="distance", kind="strip", jitter=0.3, palette="viridis", hue="prompt_number")
    g.set(xlabel=None, xticklabels=[])
    g._legend.remove()
    plt.title("Distance between VLLM and HF \n Stories dataset, Completions below 112 doc length removed")
    g.savefig(path.parents[0] / "vllm_hf_distance_stories.png", dpi=600)

if __name__ == "__main__":
    main()
    