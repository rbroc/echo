'''
Functions for generating prompts
'''
from data import load_file
def get_task_prompt(dataset:str, prompt_number:int): 
    '''
    Retrieve task prompt from specified dataset

    Args
        dataset: choose between ["dailymail_cnn", "stories", "mrpc", "dailydialog"]
        prompt_number: number from 1 and upwards. Yields ValueError if prompt_number does not exist.

    Returns
        prompt: prompt for the particular dataset and prompt number. 
    '''

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

    valid_datasets = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

    if dataset not in valid_datasets:
        valid_datasets = ", ".join(valid_datasets)
        raise ValueError(f"Invalid dataset '{dataset}'. Choose from {valid_datasets}")

    # get prompt 
    key = f"{dataset}_{prompt_number}"
    prompt = prompts.get(key, None)

    if prompt is None:
        raise ValueError(f"Invalid prompt number '{prompt_number}' for dataset '{dataset}'")

    return prompt

def get_system_prompt(chosen_model):
    system_prompts = {
            "beluga": "You are StableBeluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n",
            "llama2_chat": "You are an AI, but you do not deviate from the task prompt and you do not small talk. Never begin your response with 'Sure, here is my response: ' or anything of the like. It is important that you finish without getting cut off."
        }

    # extract

def add_task_prompt(df, dataset="stories", prompt_number=1):
    '''
    Create a task prompt without system message and add to original df.
    '''
    tp = get_task_prompt(dataset, prompt_number)

    # add task prompt to source txt 
    df[f"prompt_{prompt_number}"] = tp + df["source"].copy()

    return df 

def add_system_prompt(df, dataset="stories", prompt_number=1):
    '''
    Add system prompt for beluga and llama2chat 
    '''
    pass

def main():
    import pathlib
    dataset = "dailydialog"
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