### Classify
This folder contains all code related that make the `results/classify` happen i.e., running PCA and various classifiers.

### Overview
Code in this folder:
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| `/table` | Contains the script `create_table.py` to create tables from classifcation reports saved in `results/classify/clf_reports` that can be executed by running `run.sh` within the folder  |
| `utils/classify.py` | Classify pipeline functions |
| `run_clf_all_features.py` | CLF on all PC components |
| `run_clf_top_features.py` | CLF on the top N PC components |
| `run_clf_tfidf.py` | CLF on TF-IDF vectorised completions |
| `run_clf_embeddings.py` | CLF on embeddings of completions |
| `run_llm_detector.py` | Run LLM detector on text completions (not run yet) |

### Classification Pipeline
The pipeline consists of classifying human and machine-generated texts in each domain seperately. We also group labels as such:
1. Human vs. [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
2. Human vs. [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
3. Human vs. [stabilityai/StableBeluga-7B](https://huggingface.co/stabilityai/StableBeluga-7B)
4. Human vs. [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
5. Human vs. ALL of the above models 


We do XGBOOST classification on:
1. All PC components 
2. Top N PC components (currently top 3)
3. TF-IDF vectorised data (with max features) 
4. Embeddings of completions

And LLM detection on:
5. text completions with a LLM detector

## Usage
To run all pipelines, you can simply run (from root):
```bash
bash src/classify/run.sh
```

Note that this does not run the LLM detector: 
```bash
python src/classify/run_llm_detector.py
```
(Remember to activate virtual env)

Furthermore, to re-create the tables, you can run: 
```bash
python src/classify/table.py -d {DATASET_NAME} -t {TEMPERATURE}
```
Note you have to run this for each dataset and temperature individually with arguments e.g., 

```bash
python src/classify/table.py -d mrpc -t 1
```

## Results (Prelim)
As of 7th of November, some prelim results on VALIDATION data only (test data not in use at all).

1. [dailydialog](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/classify/clf_results/clf_reports/dailydialog_temp1/all_results.html)
2. [stories](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/classify/clf_results/clf_reports/stories_temp1/all_results.html)
3. [mrpc](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/classify/clf_results/clf_reports/mrpc_temp1/all_results.html)
4. [dailymail_cnn](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/classify/clf_results/clf_reports/dailymail_cnn_temp1/all_results.html)
