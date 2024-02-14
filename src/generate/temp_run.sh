source ../../env/bin/activate

# change directory 
cd "$(dirname "$0")" 

echo -e "[INFO:] RUNNING BELUGA ..."
python run_pipeline.py -mdl "beluga7b" -prompt_n 22 -d "stories"

echo -e "[INFO:] RUNNING LLAMA ..."
python run_pipeline.py -mdl "llama2chat_13b" -prompt_n 22 -d "stories"

echo -e "[INFO:] RUNNING MISTRAL ..."
python run_pipeline.py -mdl "mistral7b" -prompt_n 22 -d "stories"

deactivate