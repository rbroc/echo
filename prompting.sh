#!/usr/bin/env bash

# activate env 
source ./env/bin/activate

# Define the datasets, models, and prompts
datasets=("dailydialog" "dailymail_cnn" "mrpc" "stories") 
models=("beluga", "llama2_chat")
prompts=(1 2 3 4 5 6)

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for prompt in "${prompts[@]}"
        do
            echo "Processing dataset: $dataset with model: $model and prompt: $prompt"
            python src/generate/run_pipeline.py --filename "$dataset" --chosen_model "$model" --prompt_number "$prompt" --data_subset 100 --batch_size 10
        done
    done
done

deactivate
