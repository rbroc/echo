"""
Run embeddings on data parquet file and save the embeddings in a numpy file: 

python src/create_embeddings.py --in_file /path/to/data.parquet --out_file /path/to/embeddings.npy
"""

import argparse
import json
import pathlib

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for encoding",
        default=16,
    )

    parser.add_argument(
        "-t", "--temp", default=1, help="Temperature of generations", type=float
    )

    parser.add_argument("--in_file", type=str, help="Path to file with data to encode")
    parser.add_argument("--out_file", type=str, help="Path to save the embeddings")

    args = parser.parse_args()
    return args


def main():
    args = input_parse()
    temp = args.temp

    if (
        temp == 1.0
    ):  # when temp is specified from CLI, it is a float (1.0), so needs to be converted (while allowing for 1.5 as input)
        temp = int(temp)

    path = pathlib.Path(__file__)
    cache_models_path = path.parents[3] / "models"

    splits = ["train", "val", "test"]

    # load model
    model = SentenceTransformer(
        model_name_or_path="nvidia/NV-Embed-v2", # as of 2024-28-10, the best model on MTEB (classification)
        trust_remote_code=True,
        cache_folder=cache_models_path,
    )

    # switch to FP16 (as FP32 caused memory issues) - see https://github.com/UKPLab/sentence-transformers/issues/822 
    model.half()

    # load and process splits
    for split in splits:
        print(f"[INFO:] Embedding text from {split} split")
        default_paths = {
            "in_file": path.parents[2]
            / "datasets_complete"
            / "text"
            / f"temp_{temp}"
            / f"{split}_text.parquet",
            "out_file": path.parents[2]
            / "datasets_complete"
            / "embeddings"
            / f"temp_{temp}"
            / f"{split}_embeddings.npy",
        }

        # grab data from default path if not provided
        in_file = (
            pathlib.Path(args.in_file)
            if args.in_file is not None
            else default_paths["in_file"]
        )

        # read data, extract sents
        df = pd.read_parquet(in_file)
        sents = df["completions"].tolist()

        embeddings = model.encode(
            sentences=sents,
            batch_size=args.batch_size,
            show_progress_bar=True,
        )

        # save embeddings
        out_file = (
            pathlib.Path(args.out_file)
            if args.out_file is not None
            else default_paths["out_file"]
        )

        # create directory
        out_file.parents[0].mkdir(parents=True, exist_ok=True)

        np.save(out_file, embeddings)


if __name__ == "__main__":
    main()
