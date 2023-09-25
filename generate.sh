#!/usr/bin/env bash

source ./env/bin/activate

#datasets=("dailydialog" "dailymail_cnn" "mrpc" "stories") 

# Define the datasets, models, and prompts
datasets=("dailymail_cnn" "stories")
models=("falcon_instruct")
prompts=(1 2 3 4 5 6)

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for prompt in "${prompts[@]}"
        do
            echo "Processing dataset: $dataset with model: $model and prompt: $prompt"
            python src/gen_pipeline.py --filename "$dataset" --chosen_model "$model" --prompt_number "$prompt" --data_subset 150 --batch_size 10
        done
    done
done

deactivate
