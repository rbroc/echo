# ECHO 
<p align="center">
A Scalable and Explainable Approach to Discriminating Between Human and Artificially Generated Text
</p>

## Table of Contents  
1. [Road Map](#road-map)
2. [Repository Overview](#repository-overview)  
3. [Usage](#Usage)
4. [Datasets Overview](#datasets-overview)
5. [Models](#models)

## Road Map 
Refer to the project description [here](https://cc.au.dk/en/clai/current-projects/a-scalable-and-explainable-approach-to-discriminating-between-human-and-artificially-generated-text) for more detailed information.

1. 🚀 **Prompting** (Completed)

2. 📈 **Generating Data at Large Scale** (Completed)

3. 📊 **Extracting Metrics** (Completed)

4. 🤖 **Training Classifiers** (In progress)

5. 🧪 **Experimental Design** (Upcoming)

## Repository Overview
The main contents of the repository is listed below.
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| `datasets` | Original datasets `human_datasets` which are described in the [overview](#datasets-overview) below and the generated `ai_datasets`.        |
| `src` |  Scripts for generating data, running PCA/computing distances, and extracting metrics. See `src/README.md` for greater detail. |
| `results` | Preliminary results (distance plots, length distributions etc.)  |
| `metrics` | Text metrics for each dataset (human and ai), extracted with [textdescriptives](https://hlasse.github.io/TextDescriptives/)         |
| `notes` | Jupyter notebooks used for meetings with the `echo` team to present progress |
| `tokens` |Place your `.txt` token here for the HuggingFace Hub to run `llama2` models.|

## Usage 
The setup was tested on Ubuntu **22.04** (UCloud, Coder Python **1.87.2**) using Python **3.10.12**. 

### Setup 
To install necessary requirements in a virtual environment (`env`), please run the `setup.sh` in the terminal:
```bash
bash setup.sh
```
### Generating Text 
To reproduce the generation of text implemented with `vLLM`, run in the terminal:
```
bash src/generate/run.sh
```
Note that this will run several models on all datasets for various temperatures.

If you wish to play around with individual models/datasets or use the Hugging Face `pipeline` implementation, please refer to the instructions in [src/generate/README.md](/src/generate/README.md).

### Running Other Parts of the Pipeline
To run other parts of the pipeline such as analysis or cleaning of data, please refer to the individual subfolders and their readmes. For instance, the `src/metrics/README.md`. 

## Datasets Overview
All datasets can be found under `datasets/human_datasets`

In each folder, `data.ndjson` contains the processed version of the dataset (lowercased).
Each folder also contains additional files, used e.g., to generate or inspect the datasets. <br>
Our datasets are sampled from the following datasets:
- `dailymail_cnn`: https://huggingface.co/datasets/cnn_dailymail. This is a summarization dataset, which includes both extractive and abstractive summarization. Currently, 3000 examples have been sampled;
- `dailydialog`: https://huggingface.co/datasets/daily_dialog. Dialog dataset. We sampled n-1 turns as context, and the last turn is tagged as human completion. Currently, 5000 examples have been sampled, with varying context length. This dataset also includes manual emotion and speech act annotations for both context and completions;
- `mrpc`: https://paperswithcode.com/dataset/mrpc. Paraphrase corpus, from which we extract only examples that are manually labelled as paraphrases. Currently, we have 3900 examples;
- `stories`: prompts and completions for story generation. The dataset is described here: https://aclanthology.org/P18-1082/. Currently, we have 5000 examples.

`README` files within each folder include further details for each dataset.

### Preprocessing
For `dailydialog`, punctuation has been standardized and irregular transcription has been normalized (see `datasets/dailydialog/utils.py`).
Text for all dataset is lowercased, but further preprocessing may be needed.
Unprocessed datasets are kept under `datasets/*/raw.ndjson`.

## Models 
The currently used models for data generation (as per 19th March 2024):
1. llama-chat 7b ([meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf))
2. beluga 7b ([stabilityai/StableBeluga-7B](https://huggingface.co/stabilityai/StableBeluga-7B))
3. mistral 7b ([mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf))
4. llama-chat 13b ([meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf))