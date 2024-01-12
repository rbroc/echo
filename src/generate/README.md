## Overview
This folder contains the following files: 
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| `models.py` | Classes to intialize the LLMs used. Currently supports StableBeluga and Llama-chat models (quantized and full)|
| `prompts.py` | Functions to create task and system prompts. 
| `run_pipeline.py` | Generate text! Relies on `models.py` and `prompts.py`|

⚠️ NOTE that only `run_pipeline.py` can be run in the terminal!

## Generating Text 
To run a default generation text pipeline, run in the terminal (from root):
```
bash generate.sh
```
This runs [stabilityai/StableBeluga-7B](https://huggingface.co/stabilityai/StableBeluga-7B) (referred to as `beluga7b`) on the `stories` dataset. 

### Custom Generation Pipeline
To run a custom pipeline, run in the terminal (from root): 
```
python src/generate/run_pipeline.py -mdl {MODELNAME} -d {DATASETNAME} -prompt_n {PROMPT_NUMBER} -subset {DATA_SUBSET}
```

Arguments:


| Argument     | Description                                                                      | Default                |
|:-------------|:---------------------------------------------------------------------------------|:-----------------------|
| `-d`         | Name of dataset. Options: `dailydialog`, `dailymail_cnn`, `mrpc`, `stories`.     | `stories`              |
| `-mdl`       | Model name (shortened). See [models.py](models.py) for overview.                 | `beluga7b`             |
| `-prompt_n`  | Integer between 1 and 6. See [prompts.py](prompts.py) for details.                | `1`                    |
| `-subset`    | Integer specifying subset size of the data `dataset[:subset]`.                   | `None` (no subset)     |
| `-batch`     | Batch size for dataset processing, mainly for parallel GPU processing.           | `1` (no batching)      |


## Technical Requirements
Note that models require a lot of compute power and should be run on a GPU. The smaller 7b models can be run on a 32/64 CPU machine on a small subset especially if `stories` is avoided. The quantized 70b models cannot be run without a GPU.