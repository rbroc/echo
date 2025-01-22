#!/bin/bash

# get the dir 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# venv 
source "$SCRIPT_DIR/../../env/bin/activate"

# change script dir 
cd "$SCRIPT_DIR"

echo "[INFO:] IDENTIFYING NA METRICS"
python identify_NA_metrics.py

# run pca 
python run_pca.py

# deactivate venv
deactivate