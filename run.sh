#!/usr/bin/env bash

source ./env/bin/activate

# run models sequentially
for dataset in stories
do
    echo $dataset
    python src/extract_metrics.py --input $dataset
done

# close venv
deactivate

# dailydialog dailymail_cnn mrpc 
