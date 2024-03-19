# get the dir 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# venv 
source "$SCRIPT_DIR/../../env/bin/activate"

# Define the datasets, models, and prompts
datasets=("dailydialog" "dailymail_cnn" "mrpc" "stories") 
models=("beluga7b" "llama2_chat13b")
prompts=(1 2 3 4 5 6)

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for prompt in "${prompts[@]}"
        do
            echo "Processing dataset: $dataset with model: $model and prompt: $prompt"
            python generate/run_pipeline.py --filename "$dataset" --chosen_model "$model" --prompt_number "$prompt" --data_subset 100 --batch_size 10 -hf
        done
    done
done

deactivate
