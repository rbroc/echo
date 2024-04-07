## Extracting Metrics 
You can extract metrics from ALL data (`human` and `ai`) by running either of the two commands in the terminal (depending on current directory):

```bash
# from subfolder (cd src/metrics)
bash run.sh

# from root 
bash src/metrics/run.sh
```
### Extracting Metrics
If you wish to run the extraction for a particular dataset or for either human or AI, you can specify custom args and run `extract_metrics.py`: 
```bash
# from root (can also be run from subfolder)
python src/metrics/extract_metrics.py -d {DATASET NAME} -human_only {BOOL} -ai_only {BOOL}
```
Note the flags `-human_only` and `-ai_only` can be added to process solely human or ai text, respectively. 

The valid datasets for the `-d` arguments are `stories`, `dailydialog` `dailymail_cnn` and `mrpc`. 