'''
Functions for generating prompts
'''
import pandas as pd

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

def add_prompts_to_df(model, df, dataset:str="stories", prompt_number=1):
    '''
    Add formatted prompts to the DataFrame.

    Args:
        model: an instance of the Model class from Models.py
        dataset: name of dataset 
        df: The DataFrame to which prompts are to be added.
        task_prompt: The task-specific prompt.
        prompt_number: The prompt number (default is 1).
    '''
    # get task prompt
    task_prompt = get_task_prompt(dataset=dataset, prompt_number=prompt_number)

    formatted_prompts = []

    for row in df.itertuples():
        source_text = row.source

         # define user prompt (task prompt + text e.g., ""summarize this: 'I love language models'")
        user_prompt = task_prompt + source_text

        # extract formatted prompt with model (if model has not system prompt, this step will do nothing)
        final_prompt = model.format_prompt(user_prompt)

        formatted_prompts.append(final_prompt)

    df[f"prompt_{prompt_number}"] = formatted_prompts

    return df