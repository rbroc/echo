'''
Prompting functions 
'''

class PromptGenerator:
    def __init__(self, prompt_number=1):
        self.prompt_number = prompt_number

    def get_prompt(self, datafile):
        prompts = {
            # daily mail (summarization)
            "dailymail_cnn_1": "summarize the main points of this article: ",
            "dailymail_cnn_2": "create a summary of the news article: ",
            "dailymail_cnn_3": "write a short summarized text of the news article: ",
            "dailymail_cnn_4": "summarize this: ",
            "dailymail_cnn_5" : "What are the important parts of this article?: ",

            # stories (text generate)
            "stories_1": "continue the story: ",
            "stories_2": "write a small text based on this story: ",
            "stories_3": "complete the text: ",
            "stories_4": "complete the story: ",
            "stories_5": "finish the story: ",
            "stories_6": "Make a story based on this writing prompt : ",

            # mrpc (paraphrase)
            "mrpc_1": "paraphrase this text: ",
            "mrpc_2": "summarize this text: ",
            "mrpc_3": "summarize this: ",
            "mrpc_4": "create a summary of this: ",

            # dailydialog
            "dailydialog_1": "respond to the final sentence: ",
            "dailydialog_2": "continue this dialog: ",
            "dailydialog_3": "finish this dialog: ",
            "dailydialog_4": "respond to the final sentence: ",
            "dailydialog_5": "continue writing the next sentence in this: "
        }

        # returns the prompt that corresponds to the prompt number it was initalised with and the datafile specified in get .get_prompt() method 
        return prompts.get(f"{datafile}_{self.prompt_number}", "")

    def create_prompt(self, df, datafile="dailymail_cnn"):
        # retrieve prompt
        prompt = self.get_prompt(datafile)

        # if there is a prompt (valid datafile and valid prompt number), add this prompt to the source text 
        if prompt:
            df[f"prompt_{self.prompt_number}"] = prompt + df["source"].copy()
            return df

        else:
            print(f"No prompt created for datafile '{datafile}'. Invalid prompt_number or no prompts available.")
            return df

class BelugaPromptGenerator(PromptGenerator):
    def create_beluga_prompt(self, df, datafile="dailymail_cnn"):
        '''
        Create prompt column for Stable Beluga 7B 
        (the model follows a specific prompt structure: https://huggingface.co/stabilityai/StableBeluga-7B) 
        '''
        # define system prompt
        system_prompt = (
            "### System:\nYou are StableBeluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"
        )
        
        # retrieve task prompt 
        task_prompt = self.get_prompt(datafile)

        # empty list for beluga formatted prompts
        formatted_prompts = []

        for row in df.itertuples():
            # retrieve source text
            source_text = row.source

            # if there is a task prompt, then combine task prompt and source text and format for stable beluga
            if task_prompt:
                # e.g., "summarize this text: 'I love language models'"
                user_prompt = task_prompt + source_text

                # final, formatted prompt e.g., "### System:\nYou are StableBeluga (...) ### User: summarize this text: 'I love to work with language models' \n\n### Assistant:\n"
                final_prompt = f"{system_prompt}### User: {user_prompt}\n\n### Assistant:\n"

                # append to formatted prompts list 
                formatted_prompts.append(final_prompt)

            else:
                formatted_prompts.append(f"No prompt created for datafile '{datafile}'. Invalid prompt_number or no prompts available.")

        # add formatted prompts to df 
        df[f"prompt_{self.prompt_number}"] = formatted_prompts

        return df