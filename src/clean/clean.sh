#!/usr/bin/env bash

source env/bin/activate

# change directory 
cd "$(dirname "$0")" 

echo -e "[INFO:] CLEANING ..."
python clean_data.py

echo -e "[INFO:] INSPECTING ..."
python inspect_data.py

deactivate