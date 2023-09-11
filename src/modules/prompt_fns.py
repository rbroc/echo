'''
Classes for generating prompts. Contains the special class "SpecialPromptGenerator" for embedding task prompts within an interface (used for chat-models "llama2_chat" and "beluga"). 
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
            "dailymail_cnn_6" : "write highlights for this article: ",

            # stories (text generate)
            "stories_1": "continue the story: ",
            "stories_2": "write a small text based on this story: ",
            "stories_3": "complete the text: ",
            "stories_4": "complete the story: ",
            "stories_5": "finish the story: ",
            "stories_6": "Make a story based on this writing prompt : ",

            # mrpc (paraphrase)
            "mrpc_1": "paraphrase this text: ",
            "mrpc_2": "generate a sentence with a similar meaning: ",
            "mrpc_3": "paraphrase this: ",
            "mrpc_4": "make a sentence that means the same as this",

            # dailydialog
            "dailydialog_1": "respond to the final sentence: ",
            "dailydialog_2": "continue this dialog: ",
            "dailydialog_3": "finish this dialog: ",
            "dailydialog_4": "continue writing the next sentence in this: "
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

class SpecialPromptGenerator(PromptGenerator):
    '''
    Create special prompts for models StableBeluga and LLama2 chat versions
        StableBeluga follows this prompt structure: https://huggingface.co/stabilityai/StableBeluga-7B) 
        Llama2 chat versions follow this prompt structure: https://gpus.llm-utils.org/llama-2-prompt-template/ (note that the DEFAULT system prompt is recommended to be removed https://github.com/facebookresearch/llama/commit/a971c41bde81d74f98bc2c2c451da235f1f1d37c. Custom system prompts may be useful. Regardless of whether a system prompt is used, the format below is required to produce intelligble text) 
    '''

    def __init__(self, prompt_number, model_type): 
        # inherit prompt number from super class, define model type 
        super().__init__(prompt_number)
        self.model_type = model_type

    def get_system_prompt(self, model_type):
        system_prompts = {
            "beluga": "You are StableBeluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n",
            "llama2_chat": "You are an AI, but you do not deviate from the task prompt and you do not small talk. You get straight to the point."
        }
    
        return system_prompts.get(model_type, "")

    def create_prompt(self, df, datafile="dailymail_cnn"):
        # retrieve system prompts
        system_prompt = self.get_system_prompt(self.model_type)

        # retrieve task prompt
        task_prompt = self.get_prompt(datafile)

        # empty list for formatted prompts
        formatted_prompts = []
        
        # iterate over dataframe
        for row in df.itertuples():
            # extract source text 
            source_text = row.source
            
            # format prompts IF there is a task prompt 
            if task_prompt:
                # define user prompt (task prompt e.g., ""summarize this: 'I love language models'")
                user_prompt = task_prompt + source_text
                
                # format the final prompt depending on the model type 
                if self.model_type == "beluga":
                    final_prompt = f"### System:\n{system_prompt}### User: {user_prompt}\n\n### Assistant:\n"
                
                elif self.model_type == "llama2_chat":    
                    final_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
                
                # add to list of formatted prompts
                formatted_prompts.append(final_prompt)

            else:
                formatted_prompts.append(f"No prompt created for datafile '{datafile}'. Invalid prompt_number or no task prompts available.")

        df[f"prompt_{self.prompt_number}"] = formatted_prompts

        return df
    
class OneShotGenerator(PromptGenerator):
    '''
    One shot learning generator for T5 models.
    '''
    def __init__(self, prompt_number): 
        # inherit prompt number from super class, define model type 
        super().__init__(prompt_number)

    def create_prompt(self, df, datafile="dailymail_cnn"):
        # retrieve task prompt
        task_prompt = self.get_prompt(datafile)

        # retrieve first example
        example = df.iloc[0]
        example_source = example["source"]
        example_completion = example["human_completions"] # take a human completion as example 

        # create one shot example
        one_shot_example = task_prompt + example_source + f"Completion: {example_completion}"

        formatted_prompts = []

        # iterate over dataframe
        for row in df.itertuples():
            # extract source text 
            source_text = row.source
            
            # format prompts IF there is a task prompt 
            if task_prompt:
                # define user prompt (task prompt e.g., ""summarize this: 'I love language models'")
                user_prompt = task_prompt + source_text
                
                # embed oneshot example
                final_prompt = f"Example: {one_shot_example}. User: {user_prompt}. Completion: "
                
                # add to list of formatted prompts
                formatted_prompts.append(final_prompt)

            else:
                formatted_prompts.append(f"No prompt created for datafile '{datafile}'. Invalid prompt_number or no task prompts available.")

        df[f"prompt_{self.prompt_number}"] = formatted_prompts

        return df


         




    