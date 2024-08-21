### Classify
This folder contains all code related that make the `results/classify` happen i.e., running PCA and various classifiers.

### Overview
Code in this folder:
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| `/pca` | PCA folder with the script `run_PCA.py` to run PCA prior to classify pipeline  |
| `run_clf_all_features.py` | Run XGBOOST on all PC components |
| `run_clf_tfidf.py` | Run Logistic Regression on TF-IDF vectorised completions |
| `run_clf_top_features.py` | Run XGBOOST on the top N PC components |
| `` | Run XGBOOST on the top N PC components |

### Classification Pipeline
The pipeline consists of classifying human and machine-generated texts in each domain seperately. We also group labels as such:
1. Human vs. [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
2. Human vs. [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
3. Human vs. [stabilityai/StableBeluga-7B](https://huggingface.co/stabilityai/StableBeluga-7B)
4. Human vs. [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
5. Human vs. ALL of the above models 


We furthermore (as described in the scripts overview) do three pipelines: 
1. XGBOOST on all PC components 
2. XGBOOST on the top N PC components (currently top 3)
3. Logistic regression on TF-IDF vectorised data (with max features) 

## Usage
To run all three pipelines, you can simply run (from root):
```bash
bash src/classify/run.sh
```

Furthermore, to re-create the tables, you can run: 
```python
python src/classify/table.py -d {DATASET_NAME} -t {TEMPERATURE}
```
Note you have to run this for each dataset and temperature individually with arguments e.g., 

```python
python src/classify/table.py -d mrpc -t 1
```

## Results (Prelim)
As of 20th of August, some prelim results on VALIDATION data only (test data not in use at all). The reason for it being preliminary is that some final decisions have to be made still (see PR [#66](https://github.com/rbroc/echo/pull/66))

1. [dailydialog](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/classify/clf_results/clf_reports/dailydialog_temp1/all_results.html)
2. [stories](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/classify/clf_results/clf_reports/stories_temp1/all_results.html)
3. [mrpc](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/classify/clf_results/clf_reports/mrpc_temp1/all_results.html)
4. [dailymail_cnn](https://htmlpreview.github.io/?https://github.com/rbroc/echo/blob/main/results/classify/clf_results/clf_reports/dailymail_cnn_temp1/all_results.html)
