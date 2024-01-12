#!/usr/bin/env bash

source ./env/bin/activate

echo "INITIALISING SCRIPT ..."
python src/generate/run_pipeline.py --dataset "stories" --model_name "beluga7b" --prompt_number 1 --batch_size 40 # NB note prompt number!!! 

deactivate