#!/usr/bin/bash

source ./env/bin/activate

# run models sequentially
python src/extract_metrics.py --input dailymail_cnn.json
python src/extract_metrics.py --input stories_5bins.json
python src/extract_metrics.py --input mrsp_extracted.json
python src/extract_metrics.py --input dailydialog/data.json

# close venv
deactivate

