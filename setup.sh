#!/bin/bash

python3.11 -m venv env
source ./env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
deactivate
