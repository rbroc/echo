import pandas as pd
import pathlib

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

    ai_paths = sorted(ai_paths)

    return ai_paths
def main():
    path = pathlib.Path(__file__)

    for dataset in DATASETS:
        ai_dir = path.parents[1] / "datasets_files" / "text" / "ai_datasets" / "clean_data"
        human_file = path.parents[1] / "datasets_files" / "text" / "human_datasets" / dataset / "data.ndjson"

        ai_paths = get_ai_paths(ai_dir=ai_dir, dataset=dataset, temp=1)


        for p in ai_paths:
            df = pd.read_json(p, lines=True)            
            completions = df["completions"].tolist()

            # save to txt file in new 
            txt_dir = path.parents[1] / "datasets_files" / "mda" / "ai_datasets" / "txt"
            txt_file = txt_dir / p.parents[0].name / f"{p.stem}.txt" # dir / model_name / file_name.txt
            txt_file.parent.mkdir(parents=True, exist_ok=True)

            with open(txt_file, "w") as f:
                for i, completion in enumerate(completions, start=1):
                    # save each completion with a delimiter
                    f.write(f"{i}\t{completion}\n")


        # load human data
        human_df = pd.read_json(human_file, lines=True)
        human_completions = human_df["human_completions"].tolist()

        # save to txt file in 
        txt_dir = path.parents[1] / "datasets_files" / "mda" / "human_datasets" / "txt"
        txt_file = txt_dir / f"{dataset}.txt"
        txt_file.parent.mkdir(parents=True, exist_ok=True)

        with open(txt_file, "w") as f:
            for i, completion in enumerate(human_completions, start=1):
                # save each completion with a delimiter
                f.write(f"{i}\t{completion}\n")

if __name__ == "__main__":
    main()