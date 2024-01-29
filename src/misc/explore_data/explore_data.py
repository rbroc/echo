'''
Script to investigate generated data
'''
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parents[3]))
from src.utils.process_generations import preprocess_datasets
from src.utils.pca import get_descriptive_metrics, run_PCA
from src.utils.distance import compute_distances

def main(): 
    path = pathlib.Path(__file__)
    ai_dir = path.parents[3] / "datasets" / "ai_datasets" / "vLLM" / "FULL_DATA"
    human_dir = path.parents[3] / "datasets" / "human_datasets"
    
    models = ["beluga7b", "llama2_chat13b", "mistral7b"]
    datasets = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

    df = preprocess_datasets(ai_dir, human_dir, models, datasets)

    # run pca
    metrics_df = get_descriptive_metrics(df, "completions", "id")
    pca, pca_df = run_PCA(df, "completions", "id")

    # run distance
    distances = compute_distances(pca_df, "completions", "id")

    # save both 
    pca_df.to_csv(path.parents[0] / "pca_df.csv")
    distances.to_csv(path.parents[0] / "distances.csv")

if __name__ == "__main__":
    main()

