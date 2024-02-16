#!/bin/bash

# get the dir 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# venv 
source "$SCRIPT_DIR/../../env/bin/activate"

# change script dir 
cd "$SCRIPT_DIR"

# datasets and models
datasets=("dailydialog" "dailymail_cnn" "mrpc" "stories") 
models=("beluga7b")
prompt=21
temp=2

# over datasets and models
for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        echo "Processing dataset: $dataset with model: $model, prompt: $prompt and temperature $temp"
        python run_pipeline.py --dataset "$dataset" --model_name "$model" --prompt_number "$prompt" --temperature "$temp"
    done
done

deactivate