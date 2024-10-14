"""
Columns to drop for PCA and classification 
"""
def get_cols_to_drop():
    # cols to drop
    type_cols = [
        "model",
        "id",
        "is_human",
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

    return cols_to_drop