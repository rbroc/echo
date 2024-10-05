"""
Run PCA on train data (metrics) and save results to /results/classify/pca_results/temp_{temp}
"""

import pathlib
import pickle
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).parents[2]))
from src.utils.pca import (
    get_loadings,
    plot_cumulative_variance,
    plot_loadings,
    run_PCA,
)


def main():
    path = pathlib.Path(__file__)

    temp = 1

    # dirs based on temp
    savedir = path.parents[2] / "results" / "classify" / "pca_results" / f"temp_{temp}"
    loadingspath = savedir / "loadings"

    datapath = (
        path.parents[2]
        / "datasets_complete"
        / "metrics"
        / f"temp_{temp}"
        / "train_metrics.parquet"
    )

    file_name = "train"

    # load train metrics data
    train_df = pd.read_parquet(datapath)

    # DEFINE FEATURES #
    all_features = train_df.columns.tolist()

    # cols to drop
    type_cols = [
        "model",
        "id",
        "is_human",
        "unique_id",
        "sample_params",
        "temperature",
        "prompt_number",
        "dataset",
        "annotations",
    ]  #     # cols that are directly tied to type of generation (and should therefore not be included in classification)
    na_cols = [
        "first_order_coherence",
        "second_order_coherence",
        "smog",
        "pos_prop_SPACE",
        "per_word_perplexity"
    ]  # cols found by running identify_NA_metrics.py
    manually_selected_cols = [
        "pos_prop_PUNCT"
    ]  # no punctuation wanted (due to manipulating them)
    cols_to_drop = type_cols + na_cols + manually_selected_cols

    # final feature list after dropping cols
    features = [feat for feat in all_features if feat not in cols_to_drop]

    # run PCA
    pca_model, pca_df = run_PCA(
        train_df,
        feature_names=features,
        n_components=len(features),
        keep_metrics_df=False,
    )  # keep_metrics_df=False to only keep pca components and row identifiers

    # cumvar
    plot_cumulative_variance(
        pca_model,
        f"{file_name.capitalize()} data (Temperature of {temp})",
        savedir,
        f"{file_name}_CUMVAR.png",
    )

    # loadings
    loadings = get_loadings(pca_model, features)
    components = loadings.columns[:5]

    for comp in components:
        plot_loadings(loadings, comp, loadingspath)

    # save results + model
    with open(savedir / f"{file_name}_model.pkl", "wb") as file:
        pickle.dump(pca_model, file)

    with open(savedir / f"{file_name}_EXPVAR.txt", "w") as file:
        file.write("PRINCIPAL COMPONENTS: EXPLAINED VARIANCE\n")
        file.write(f"Features: {features}\n")

        for i, variance in enumerate(pca_model.explained_variance_ratio_, start=1):
            file.write(f"pca_{i}: {variance:.8f}\n")

        expvar_df = pd.DataFrame(
            pca_model.explained_variance_ratio_, columns=["explained_variance"]
        )
        expvar_df.to_csv(savedir / f"{file_name}_EXPVAR.csv")


if __name__ == "__main__":
    main()
