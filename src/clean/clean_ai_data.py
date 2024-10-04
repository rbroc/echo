"""
Clean generated AI data to match human data (e.g., lowercase, removing irregular format)
"""

import ast
import pathlib
import re

import pandas as pd
from tqdm import tqdm

MODELS = ["beluga7b", "llama2_chat7b", "llama2_chat13b", "mistral7b"]
PROMPT_NUMBERS = [21]
DATASETS = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]


def get_ai_paths(
    ai_dir: pathlib.Path, dataset: str = "dailydialog", temp: float | int = 1
) -> list[pathlib.Path]:
    """
    Get all paths pertaining to a particular dataset (e.g., mrpc, dailymail_cnn, stories, dailydialog, etc.) for models
    """
    ai_paths = []

    if temp and not isinstance(temp, (int, float)):
        raise ValueError(f"Temperature must be a int or float, not {type(temp)}")
    if PROMPT_NUMBERS and not isinstance(PROMPT_NUMBERS, list):
        raise ValueError(f"Prompt number must be an list, not {type(PROMPT_NUMBERS)}")

    for model_name in MODELS:
        model_path = ai_dir / model_name

        for prompt_number in PROMPT_NUMBERS:
            file_identifier = f"{dataset}_prompt_{prompt_number}_temp{temp}.ndjson"
            file_path = model_path / file_identifier
            ai_paths.extend([file_path])

    if len(ai_paths) == 0:
        print(
            f"[WARNING:] Length of ai paths is zero. Ensure that you have valid arguments."
        )

    return ai_paths


def clean_ai_df(df, col="completions"):
    """
    lowercase, remove irregular format to standardise to human datasets like in src/clean/clean_data.py
    """
    df[col] = df[col].str.replace(r"^\s+", "", regex=True)  # rm space at beginning
    df[col] = df[col].str.lower()  # lowercase
    df[col] = df[col].str.replace(
        r"^(a:|b:)", lambda m: m.group(1).upper(), regex=True
    )  # convert a: or b: to A: or B:
    df[col] = df[col].str.replace("<newline>", " ", regex=False)  # rm newline
    df[col] = df[col].str.replace(r"\s+", " ", regex=True)  # rm extra spaces

    return df


def standardize_ai_data(ai_dfs: list[pd.DataFrame], clean: bool = True):
    """
    Standardize AI data to match human data.

    Args:
        ai_dfs: list of dataframes
        clean_ai: whether to clean ai data (e.g., lowercase, removing irregular format)

    Returns:
        ai_dfs: list of dataframes
    """
    for idx, df in enumerate(ai_dfs):
        new_df = df.copy()

        new_df["is_human"] = 0

        # standardise prompt and completions cols
        prompt_colname = [col for col in new_df.columns if col.startswith("prompt_")][
            0
        ]  # get column name that starts with prompt_ (e.g., prompt_1, prompt_2, ...)
        new_df["prompt_number"] = prompt_colname.split("_")[1]  # extract numbers
        new_df.rename(columns={prompt_colname: "prompt"}, inplace=True)

        mdl_colname = [col for col in new_df.columns if col.endswith("_completions")][0]
        new_df["model"] = re.sub(
            r"_completions$", "", mdl_colname
        )  # remove "_completions" from e.g., "beluga_completions"
        new_df.rename(columns={mdl_colname: "completions"}, inplace=True)

        # add temperature val to col
        if "sample_params" in df.columns:
            new_df["temperature"] = df.sample_params.apply(
                lambda x: (
                    ast.literal_eval(x).get("temperature") if pd.notna(x) else None
                )
            )

        if clean:
            new_df = clean_ai_df(new_df)

        ai_dfs[idx] = new_df  # replace original df with new df

    return ai_dfs


def main():
    path = pathlib.Path(__file__)
    ai_dir = (
        path.parents[2]
        / "datasets_files"
        / "text"
        / "ai_datasets"
        / "vLLM"
        / "FULL_DATA"
    )

    out_dir = path.parents[2] / "datasets_files" / "text" / "ai_datasets" / "clean_data"

    temp = 1

    for dataset in tqdm(DATASETS, desc="Datasets"):
        # get all ai paths for a particular dataset and temperature
        ai_paths = get_ai_paths(ai_dir, dataset=dataset, temp=temp)

        for p in tqdm(ai_paths, desc="AI Paths"):
            # get model name
            model_name = p.parents[0].name

            # create dir
            file_dir = out_dir / model_name
            file_dir.mkdir(parents=True, exist_ok=True)

            file_name = p.name  # e.g., dailymail_cnn_prompt_21_temp1.ndjson

            # read files, and save to "clean_data" but model name as folder name
            df = pd.read_json(p, lines=True)
            df = standardize_ai_data([df], clean=True)
            df[0].to_csv(file_dir / file_name, index=False)


if __name__ == "__main__":
    main()
