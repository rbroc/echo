### Running the code

The easiest way is to clone the repo and run the setup script to create a virtual environment and install packages:

```bash
bash setup.sh
```

You can then extract metrics from all of the data using the second bash script:

```bash
bash run.sh
```

The pipeline requries  *Python3.11* - it should work down to *3.9*, but hasn't been tested.


### Datasets
All datasets can be found under `datasets`. In each folder, `data.ndjson` contains the processed version of the dataset (lowercased).
Each folder also contains additional files, used e.g., to generate or inspect the datasets. <br>
Our datasets are sampled from the following datasets:
- `dailymail_cnn`: https://huggingface.co/datasets/cnn_dailymail. This is a summarization dataset, which includes both extractive and abstractive summarization. Currently, 3000 examples have been sampled;
- `dailydialog`: https://huggingface.co/datasets/daily_dialog. Dialog dataset. We sampled n-1 turns as context, and the last turn is tagged as human completion. Currently, 5000 examples have been sampled, with varying context length. This dataset also includes manual emotion and speech act annotations for both context and completions;
- `mrpc`: https://paperswithcode.com/dataset/mrpc. Paraphrase corpus, from which we extract only examples that are manually labelled as paraphrases. Currently, we have 3900 examples;
- `stories`: prompts and completions for story generation. The dataset is described here: https://aclanthology.org/P18-1082/. Currently, we have 5000 examples.

README files within each folder include further details for each dataset.

### Preprocessing
For `dailydialog`, punctuation has been standardized and irregular transcription has been normalized (see `datasets/dailydialog/utils.py`).
Text for all dataset is lowercased, but further preprocessing may be needed.
Unprocessed datasets are kept under `datasets/*/raw.ndjson`.

### TODO:
- [x] Describe datasets (**partially completed**)
    - Discriminate between extractive and abstractive summaries for `dailymail_cnn`
    - Normalize transcriptions / spelling for all datasets (for `dailydialog`, punctuation has been normalized -- see scripts in relevant folder --, and both source text and human completion have been lowercased for all datasets, see `src/clean_data.py`, but there seem to be weird tokens and other irregular features in other datasets)
    - Potentially upload dataset creation files
- [ ] Identify algorithms
- [ ] Sample examples
- [ ] Set up experiment
