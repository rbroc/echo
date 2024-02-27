#!/bin/bash

# get the dir 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# venv 
source "$SCRIPT_DIR/../../env/bin/activate"

# change script dir 
cd "$SCRIPT_DIR"

# datasets and models
datasets=("dailydialog" "dailymail_cnn" "mrpc" "stories") 
models=("llama2_chat7b" "llama2_chat13b" "beluga7b" "mistral7b")
prompt=21
temperatures=(1 1.5 2)

# over datasets, models, temp
for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for temp in "${temperatures[@]}"
        do  
            echo "Processing dataset: $dataset with model: $model, prompt: $prompt and temperature $temp"
            python run_pipeline.py --dataset "$dataset" --model_name "$model" --prompt_number "$prompt" --temperature "$temp"
        done
    done
done

deactivate