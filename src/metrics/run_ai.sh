# get the dir 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# venv 
source "$SCRIPT_DIR/../../env/bin/activate"

# change script dir 
cd "$SCRIPT_DIR"

# run models and datasets sequentially
datasets=("dailydialog" "dailymail_cnn" "mrpc") # stories should also be there, but currently left out
models=("beluga7b" "llama2_chat13b")

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        echo "Processing dataset: $dataset with model: $model"
        python extract_metrics_ai.py --dataset "$dataset" --model "$model"
    done
done

deactivate