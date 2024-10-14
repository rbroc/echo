#!/bin/bash

# get the dir 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# venv 
source "$SCRIPT_DIR/../../env/bin/activate"

# change script dir 
cd "$SCRIPT_DIR"

# datasets and models
datasets=("dailydialog" "dailymail_cnn" "mrpc" "stories") 
temperatures=(1)

# over datasets and temps 
echo "[INFO:] Running classification on ALL features"
for dataset in "${datasets[@]}"
do
    for temp in "${temperatures[@]}"
    do  
        python run_clf_all_features.py --dataset "$dataset" --temp "$temp"
    done
done

echo "[INFO:] Running classification on TOP features"
for dataset in "${datasets[@]}"
do
    for temp in "${temperatures[@]}"
    do  
        python run_clf_top_features.py --dataset "$dataset" --temp "$temp"
    done
done

echo "[INFO:] Running classification with TF-IDF"
for dataset in "${datasets[@]}"
do
    for temp in "${temperatures[@]}"
    do  
        python run_clf_tfidf.py --dataset "$dataset" --temp "$temp"
    done
done

echo "[INFO:] Making CLF table for all each dataset"
for dataset in "${datasets[@]}"
do
    for temp in "${temperatures[@]}"
    do  
        python table/create_table.py --dataset "$dataset" --temp "$temp"
    done
done
