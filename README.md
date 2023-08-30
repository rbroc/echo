## Setup 
To install necessary requirements in a virtual environment (`env`), please run the `setup.sh` in the terminal:
```bash
bash setup.sh
```
Note that this requires python *3.10* which is the standard python version on the UCloud CPU's. 

## Running the Code 
### Generation Pipeline 
To run the generation text pipeline, run the script with the custom arguments:
```
python src/gen_pipeline.py -mdl {MODELNAME} -f {FILENAME} -prompt_n {PROMPT_NUMBER} -subset {DATA_SUBSET}
```

With the arguments specified below: 
| <div style="width:80px">Arg</div>    | Description                             | <div style="width:120px">Default</div>    |
| :---        |:---                                                                                        |:---             |
|```-mdl```   | Model to use for generating. Choose between `beluga`, `falcon`, `falcon_instruct`, `llama2` and `llama2_chat`. Currently these are all the smaller 7B models. See full models names at [pipeline_fns.py](src/modules/pipeline_fns.py)            | `beluga`     |
|```-f```| Filename. Choose between `dailydialog`, `dailymail_cnn`, `mrpc` and `stories`  | `stories`.              |
|```-prompt_n```   | Integer between 1 and 6 currently. See [prompt_fns.py](src/modules/prompt_fns.py) for the specific prompts to choose between for each dataset.            |    `1`            |
|```-subset```   |   Integer how big of a subset of the data you want to use `dataset[:subset]``               | `None` (i.e., no subset)               |

*arguments subject to change*

Please note that these models are quite large and require 32 or 64 CPUs to run in decent time on a small subset. For running on CPU, you may want to avoid the `stories` dataset as it is a quite heavy!

### Extracting Metrics 
You can then extract metrics from all of the data using the second bash script:

```bash
bash run.sh
```
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
