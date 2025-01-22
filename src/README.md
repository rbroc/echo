The src folder contains: 

| Folder/Script            | Description                                                                                                   |
|-------------------|---------------------------------------------------------------------------------------------------------------|
| `/analysis`        | Scripts for analysis (not currently used in pipeline) |
| `/classify`        | Scripts for classifying datasets from `datasets_complete`                                                       |
| `/clean`           | Clean and inspect human datasets (`datasets_files`)                                                                             |
| `/generate`        | Entire generation pipeline here.                                                                               |
| `/make_dataset`         | Make train, val, test splits (`datasets_complete`)                                          |
| `/metrics`         | Extract metrics using test descriptives for human and AI data.                                                |
| `/make_dataset`    | PCA on `datasets_complete/metrics/train_metrics.parquet` (gotten from running scripts in `make_dataset`)                                       |
| `/utils`           | Scripts for loading, preprocessing AI data (including combining with human datasets), running PCA, and computing distances.  |
| `clean_ai_data.py`| Clean ai data prior to extracting metrics  |


### Pipeline 
The pipeline is as follows:

0. `/clean` -> clean human datasets
1. `/generate` -> generate ai datasets (saved in `datasets_files/text/ai_datasets/`)
2. `/clean_ai_data.py` -> clean ai data (saved in `datasets_files/text/ai_datasets/clean_data`)
3. `/metrics`-> extract metrics from AI and human data (saved in `datasets_files/metrics`)
4. `/make_dataset` -> make train, val, test splits for **text** + **metrics**, and compute **embeddings** from text (saved in `datasets_complete` with subdirectories `/text`, `/metrics` and `/embeddings`). 
5. `pca` -> run and save scaler + PCA model on `datasets_complete/metrics/train_metrics.parquet`
6. `classify` -> classification on train & val for both **text** (tf-idf), **metrics** (PCA'ed features), **embeddings** (bonus: LLM detector on **text**)

## Moving beyond generation
If you are not interested in regenerating data, you can skip steps 0 to 3, as `datasets_files` is already complete with both human and LLM-generated datasets.

Yu can simply go directly to `make_dataset` and run the .sh file (see `make_dataset/README.md`) to make the train, val and test splits. You should then fit the `pca` model (see `pca/README.md`), and go from there. 
