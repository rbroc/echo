#!/bin/bash

# get the dir 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# venv 
source "$SCRIPT_DIR/../../env/bin/activate"

# change script dir 
cd "$SCRIPT_DIR"

# run scripts
echo "[INFO:] Creating text dataset ..."
python text_dataset.py

echo "[INFO:] Creating metrics dataset ..."
python metrics_dataset.py

echo "[INFO:] Checking concordance in datasets ..."
python check_datasets.py

deactivate