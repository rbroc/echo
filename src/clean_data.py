'''
Streamline human datasets
'''
import ndjson
import json
import pathlib
import re

def load_raw_data(data_rootdir:pathlib.Path=pathlib.Path(__file__).parents[1] / "datasets" / "human_datasets", dataset:str="stories"):
    '''
    load raw data 
    '''
    if dataset == "stories": 
        raw_filepath = data_rootdir / dataset / "stories_5bins_1000tokens_al.json"
        with open(raw_filepath) as f:
            data = json.load(f)
    else: 
        raw_filepath = data_rootdir / dataset / "raw.ndjson"
        with open(raw_filepath) as f:
            data = ndjson.load(f)

    return data

def clean_stories(stories, data_rootdir):
    for col in ["source", "human_completions"]:
        for s in stories:
            # lowercase the text
            s[col] = s[col].lower()

            # remove patterns enclosed by square brackets
            s[col] = re.sub(r'\[[^\]]+\]', '', s[col])

            # remove all consecutive backticks (`)
            s[col] = re.sub(r'`+', '', s[col])

            # replace multiple spaces with a single space
            s[col] = re.sub(r'\s+', ' ', s[col])

            # define punctuation marks
            punctuation_marks = r'.,!?;:'

            # adjust spaces around punctuation marks, considering contractions
            for mark in punctuation_marks:
                if mark == "'":
                    continue  # skip apostrophe to handle contractions separately
                s[col] = re.sub(r'\s*(' + re.escape(mark) + r')\s*', r'\1 ', s[col])

            # handle spaces around contractions (apostrophes)
            s[col] = re.sub(r'\s*’\s*', r'’', s[col])

            # remove space before the apostrophe in contractions
            s[col] = re.sub(r'\s+(?=\')', '', s[col])

            # handle spaces inside contractions like "doesn't"
            s[col] = re.sub(r'\s+(?=\w+\'\w+)', '', s[col])

            # remove extra space before the last quotation mark
            s[col] = s[col].strip()

    # save 
    savepath = data_rootdir / "stories" / "data.ndjson"

    with open(savepath, 'w') as f:
        ndjson.dump(stories, f, ensure_ascii=False)

    return stories

def clean_mrpc(mrpc, data_rootdir):
    for m in mrpc:
        m['human_completions'] = m['human_completions'][0][0].lower()
        m['source'] = m['source'].lower()
        m['id'] = 'mrpc' + m['id'][4:]

    # save 
    savepath = data_rootdir / "mrpc" / "data.ndjson"

    with open(savepath, 'w') as f:
        ndjson.dump(mrpc, f, ensure_ascii=False)

    return mrpc

def clean_dailymail_cnn(dailymail_cnn, data_rootdir):
    for d in dailymail_cnn:
        d['human_completions'] = d['human_completions'].lower()
        d['source'] = d['source'].lower()
    
    # save 
    savepath = data_rootdir / "dailymail_cnn" / "data.ndjson"
    
    with open(savepath, 'w') as f:
        ndjson.dump(dailymail_cnn, f, ensure_ascii=False)

    return dailymail_cnn

def replace_eot_with_speakers(text):
    '''
    Replace EOT with speaker 1 and speaker 2: 

    (for dailydialog)
    '''
    # alternate between speaker 1 and 2
    def speaker_replacer(match):
        speaker_replacer.counter = (speaker_replacer.counter + 1) % 2
        return f"speaker {speaker_replacer.counter + 1}:"

    speaker_replacer.counter = 0
    return re.sub(r'\[EOT\]', speaker_replacer, text)

def clean_dailydialog(dailydialog, data_rootdir):
    for col in ["source", "human_completions"]:
        for d in dailydialog:
            # replace EOT token with speaker 1 and 2
            d[col] = replace_eot_with_speakers(d[col])
    
    # save 
    savepath = data_rootdir / "dailydialog" / "data.ndjson"

    with open(savepath, 'w') as f:
        ndjson.dump(dailydialog, f, ensure_ascii=False)

    return dailydialog

def cleanup(datasets:list=["mrpc", "stories", "dailydialog", "dailymail_cnn"]):
    '''
    Standardize datasets by lowercasing and removing irregular format 

    Cleans and saves the following datasets:
        mrpc
        stories
        dailydialog
        dailymail_cnn
    '''
    # load raw data
    path = pathlib.Path(__file__)
    data_rootdir = path.parents[1] / "datasets" / "human_datasets"

    cleaning_functions = {
        "mrpc": clean_mrpc,
        "stories": clean_stories,
        "dailydialog": clean_dailydialog,
        "dailymail_cnn": clean_dailymail_cnn
    }

    # check for invalid dataset names 
    invalid_datasets = [d for d in datasets if d not in cleaning_functions.keys()]

    if invalid_datasets:
        raise ValueError(f"Invalid dataset names: {invalid_datasets}. Choose from {list(cleaning_functions.keys())}")

    # load and clean
    path = pathlib.Path(__file__)
    data_rootdir = path.parents[1] / "datasets" / "human_datasets"
    cleaned_data = {}

    for dataset in datasets:
        data = load_raw_data(data_rootdir, dataset)
        cleaned_data[dataset] = cleaning_functions[dataset](data, data_rootdir)

    return cleaned_data

if __name__ == '__main__':
    cleanup()