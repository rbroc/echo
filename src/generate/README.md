## Introduction
This folder contains all scripts related to generating data ([ai_datasets](../../datasets/ai_datasets)). 

The default implementation of the LLMs uses the [vLLM](https://github.com/vllm-project/vllm) library due to its high inference speeds. However, the same models can optionally be run using Hugging Face's own implementation (see [custom generation pipeline](#custom-generation-pipeline)).

## Overview
The files within this folder are:
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| `generation.py` | Functions for generation pipelines (vLLM and HF implementation). |
| `models.py` | Classes to intialize the LLMs used. Currently supports StableBeluga and Llama-chat models (quantized and full) and Mistral|
| `prompts.py` | Functions to create task prompts and format task and system prompts according to a model class
| `run_pipeline.py` | Generate text! Relies on aforementioned scripts.|

⚠️ NOTE that only `run_pipeline.py` can be run in the terminal!

## Generating Text 
To run a default generation text pipeline, run in the terminal (from root):
```
bash generate.sh
```
This runs [stabilityai/StableBeluga-7B](https://huggingface.co/stabilityai/StableBeluga-7B) (referred to as `beluga7b`) on the `stories` dataset implemented with `vLLM`.

### Custom Generation Pipeline
To run a custom pipeline, run in the terminal (from root and with `env` active): 
```
python src/generate/run_pipeline.py -mdl {MODELNAME} -d {DATASETNAME} -prompt_n {PROMPT_NUMBER} -subset {DATA_SUBSET} 
```




| Argument     | Description                                                                      | Default                |
|:-------------|:---------------------------------------------------------------------------------|:-----------------------|
| `-d`         | Name of dataset. Options: `dailydialog`, `dailymail_cnn`, `mrpc`, `stories`.     | `stories`              |
| `-mdl`       | Model name (shortened). See [models.py](models.py) for overview.                 | `beluga7b`             |
| `-prompt_n`  | Integer between 1 and 6. See [prompts.py](prompts.py) for details.                | `1`                    |
| `-subset`    | Integer specifying subset size of the data `dataset[:subset]`.                   | `None` (no subset)     |
| `-batch`     | Batch size for dataset processing, mainly for parallel GPU processing.           | `1` (no batching)      |
| `-hf`     | Bool. If specified, model will run a  the model should be run using a Hugging Face implementation         | `1` (no batching)      |

Note the additional `-batch` and `-hf` that are only relevant if you wish to run a pipeline using Hugging Face's own implementation:
```
python src/generate/run_pipeline.py -batch {BATCH_SIZE} -hf
```

## Technical Requirements
Note that models require a lot of compute power. Using the default `vLLM` implementation, GPU is required. Smaller 7B models can be run with `HF` implementation on the right CPU (32/64 node on UCloud) if a small subset is selected and `stories` is avoided. The HF implementation still requires GPU for the 70b models (quantized and full). 