#!/usr/bin/env bash

source ./env/bin/activate

# run models and datasets sequentially
datasets=("dailydialog" "dailymail_cnn" "mrpc") # stories should also be there, but currently left out
models=("beluga" "falcon_instruct" "llama2_chat")

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        echo "Processing dataset: $dataset with model: $model"
        python src/metrics/extract_metrics_ai.py --dataset "$dataset" --model "$model"
    done
done

deactivate