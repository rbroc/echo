#!/usr/bin/env bash

source ./env/bin/activate

echo "INITIALISING SCRIPT ..."
python src/generate/run_pipeline.py --filename "stories" --chosen_model "beluga7b" --prompt_number 3 --batch_size 40 #NB note prompt number!!! 

deactivate