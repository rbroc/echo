#!/usr/bin/env bash

python3 -m venv env
source ./env/bin/activate

echo "[INFO]: Installing necessary reqs in env" 
pip install -r requirements.txt

# for text descriptives 
python -m spacy download en_core_web_md

deactivate
echo "[INFO]: Done!" 