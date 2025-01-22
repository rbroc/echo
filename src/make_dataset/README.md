## Introduction
This folder contains scripts that will process the many files in `datasets_files` to `datasets_complete`. 

### Train, Val Test Splits For Text, Metrics and Embeddings
Concretely, three types of datasets will be created, split into three splits (train, test, val): 
1. `datasets_complete/text` (text generations)
2. `datasets_complete/metrics` (metrics extracted from text in `src/metrics`)
3. `datasets_complete/embeddings` (embeddings extraced from `datasets_complete/text`) 

These complete files contain all four dataset domains (e.g., "stories", "dailydiaolog"), but can later be split again during classification. 

The reason for doing so is to ensure a more streamlined process for processing and loading data, **keeping the amount of files to be worked on to a minimum** after this stage.

## Overview
The files within this folder are:
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| `text_dataset.py` | Create  text train/val/test splits from `datasets_files/text`. Includes preliminary processing to streamline data |
| `metrics_dataset.py` | Create  metrics train/val/test splits from `datasets_files/metrics` |
| `embeddings_dataset.py` | Extract embeddings from `datasets_complete/text`, creating train/test/val embeddings |
| `check_datasets.py` | Check that train/val/test splits for formats text and metrics (embeddings will not be checked, but is based on text format)|
| `util_split_data.py` | Utils script used in `text_dataset.py` and `metrics_dataset.py`|
| `run.sh` | Run dataset pipeline (create text, metrics, extract embeddings from text to create embeddings dataset)|

## Running the code 
After having installed packages via the main setup.sh ([see more](https://github.com/rbroc/echo?tab=readme-ov-file#setup)), you can type from the root directory:

```bash
bash src/make_dataset/run.sh
```

This will run the pipelines for creating all three datasets.

To save compute, the embeddings pipeline will not re-run if files already exist in `datasets_complete`. To trigger a re-run, you need to delete/move files from `datasets_complete/embeddings`.

### Running other temperatures
The run.sh only runs files for temp1, but in principle, we can run it for temperature 1.5 (if those files exist and are named accordingly):
```
python src/make_dataset/text_dataset -t 1.5
```
Note that you need to activate the virtual environment (`source/env/bin/activate` from root), created by running `setup.sh`.