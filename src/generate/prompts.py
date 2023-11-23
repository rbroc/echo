'''
Functions for generating prompts
'''
from data import load_file
def get_task_prompt(dataset, prompt_number): 
    prompts = {
        # daily mail (summarization)
        "dailymail_cnn_1": "summarize the main points of this article: ",
        "dailymail_cnn_2": "create a summary of the news article: ",
        "dailymail_cnn_3": "write a short summarized text of the news article: ",
        "dailymail_cnn_4": "summarize this: ",
        "dailymail_cnn_5" : "What are the important parts of this article?: ",
        "dailymail_cnn_6" : "write highlights for this article: ",

        # stories (text generate)
        "stories_1": "continue the story: ",
        "stories_2": "write a small text based on this story: ",
        "stories_3": "complete the text: ",
        "stories_4": "complete the story: ",
        "stories_5": "finish the story: ",
        "stories_6": "Make a story based on this writing prompt: ",

        # mrpc (paraphrase)
        "mrpc_1": "paraphrase this text: ",
        "mrpc_2": "generate a sentence with a similar meaning: ",
        "mrpc_3": "paraphrase this: ",
        "mrpc_4": "make a sentence that means the same as this: ",

        # dailydialog
        "dailydialog_1": "respond to the final sentence: ",
        "dailydialog_2": "continue this dialog: ",
        "dailydialog_3": "finish this dialog: ",
        "dailydialog_4": "continue writing the next sentence in this: "
        }

    key = f"{dataset}_{prompt_number}"
    prompt = prompts.get(key, None)

    if prompt is None:
        raise ValueError(f"Invalid dataset '{dataset}' or prompt number '{prompt_number}'")

    return prompt

def get_system_prompt():
    pass

def add_task_prompt(df, dataset="stories", prompt_number=1):
    '''
    Create a task prompt without system message and add to original df.
    '''
    tp = get_task_prompt(dataset, prompt_number)

    # add task prompt to source txt 
    df[f"prompt_{prompt_number}"] = tp + df["source"].copy()

    return df 

def main():
    import pathlib
    dataset = "stories"
    prompt_number = 2
    prompt = get_task_prompt(dataset, prompt_number)
    print(prompt)

    path = pathlib.Path(__file__)

    # load data  
    datapath = path.parents[2] / "datasets" / dataset
    datafile = datapath / "data.ndjson"
    df = load_file(datafile)

    prompt_df = add_task_prompt(df, dataset, prompt_number)

if __name__ == "__main__":
    main()