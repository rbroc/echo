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
6. [Preliminary results](#preliminary-results)

## Road Map 
Refer to the project description [here](https://cc.au.dk/en/clai/current-projects/a-scalable-and-explainable-approach-to-discriminating-between-human-and-artificially-generated-text) for more detailed information.

1. 🚀 **Prompting** (Completed)

2. 📈 **Generating Data at Large Scale** (January 2024)

3. 🧪 **Experimental Design**

4. 📊 **Extracting Metrics / Analyzing Data**

5. 🤖 **Training Classifiers**

## Repository Overview
The main contents of the repository is as such:
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| `datasets` | Contains original datasets `human_datasets` which are described in the [overview](#datasets-overview) below and the generated `ai_datasets`.        |
| `src` |  Scripts for generating data, running PCA/computing distances, and extracting metrics. |
| `results` | Contains preliminary results for prompt selection (distance plots) and a description of the results workflow. |
| `tokens` |Place your `.txt` token here for the HuggingFace Hub to run `llama2` models.|

Note that each folder has individual `READMEs` with further instructions.

## Usage 

### Setup 
To install necessary requirements in a virtual environment (`env`), please run the `setup.sh` in the terminal:
```bash
bash setup.sh
```
Note that this has been tested with python *3.10* and *3.9* which are the standard python versions on UCLOUD CPUs and GPUs, respectively. 

### Generating Text 
To run the generation text pipeline, run in the terminal:
```
bash generate.sh
```
Note that this will run text pipeline on a default dataset and model. If you wish to play around with the pipeline, see the `src/generate/README.md` for instructions.

To run the entire prompting pipeline (testing out prompts), run in the terminal:
```
bash prompting.sh
```

### Extracting Metrics 
You can then extract metrics from all of the `human data` using the second bash script:

```bash
bash run.sh
```

## Datasets Overview
All datasets can be found under `datasets/human_datasets`

In each folder, `data.ndjson` contains the processed version of the dataset (lowercased).
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

## Models 
The models that were used for prompting were the following: 
1. [stabilityai/StableBeluga-7B](https://huggingface.co/stabilityai/StableBeluga-7B) (referred to as `beluga7b`)
2. [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) (referred to as `llama2_chat13b`)

Models to be used to generate the final data will be their 70b versions (although likely quantized). 

## Preliminary Results
For preliminary results on the selection of prompts, please refer to `results/README.md` where a description of workflow and results are given.