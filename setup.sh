#!/usr/bin/env bash

python3 -m venv env
source ./env/bin/activate

echo "[INFO]: Installing necessary reqs in env" 
pip install -r requirements.txt
pip install textdescriptives==2.7.3

# upgrade for quantized mdls (if quantized mdls annoy)
#echo "[INFO]: Upgrading and installing for Quantized Mdls"
#pip install --upgrade transformers optimum 
#pip install --upgrade auto-gptq

deactivate
echo "[INFO]: Done!" 