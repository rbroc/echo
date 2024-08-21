import pathlib 
import multiprocessing as mp
import pandas as pd
import spacy

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[3]))
from src.utils.process_generations import preprocess_datasets
from src.utils.get_metrics import get_descriptive_metrics
from src.utils.pca import run_PCA, get_loadings, plot_loadings

def main(): 
    spacy.util.fix_random_seed(129)

    # paths
    path = pathlib.Path(__file__)
    ai_dir = path.parents[3] / "datasets" / "ai_datasets" / "HF" / "prompt_select"
    human_dir = path.parents[3] / "datasets" / "human_datasets"

    results_path = path.parents[3] / "results" / "prompt_select" / "PCA"
    results_path.mkdir(parents=True, exist_ok=True)

    # models     
    models = ["beluga7b", "llama2_chat13b"]
    datasets = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

    print("[INFO:] PREPROCESSING DATA, COMBINING DATAFRAMES")
    df = preprocess_datasets(ai_dir, human_dir, models, datasets, subset=99, temp=None, prompt_numbers=None)

    print("[INFO:] EXTRACTING LOW LEVEL METRICS")
    n_cores = mp.cpu_count() - 1
    metrics_df = get_descriptive_metrics(df, "completions", "en_core_web_lg", batch_size=20, n_process=n_cores)

    print("[INFO:] RUNNING PCA ...")
    pca, final_df = run_PCA(metrics_df, feature_names=["doc_length", "n_tokens", "n_characters", "n_sentences"], n_components=4)

    print(final_df)
    print(pca.explained_variance_ratio_)

    final_df.to_csv(results_path / "PCA_data.csv")

    print("[INFO:] PLOTTING PCA")
    loadings = get_loadings(pca, feature_names=["doc_length", "n_tokens", "n_characters", "n_sentences"])

    for component in loadings:
        plot_loadings(loadings, component, results_path)

if __name__ == "__main__":
    main()