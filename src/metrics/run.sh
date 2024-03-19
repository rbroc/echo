# get the dir 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# venv 
source "$SCRIPT_DIR/../../env/bin/activate"

# change script dir 
cd "$SCRIPT_DIR"

# run models sequentially
for dataset in dailydialog dailymail_cnn mrpc stories
do
    echo $dataset
    python extract_metrics.py --input $dataset
done

# close venv
deactivate

