'''
Classes and functions for generating text for the pipeline.
'''

# utils
from tqdm import tqdm

# models  
from transformers import pipeline, AutoTokenizer

# import prompting 
from modules.prompt_fns import PromptGenerator, SpecialPromptGenerator

class BaseModel():
    '''
    Base model for generating text.
    '''

    def __init__(self, chosen_model):
        self.chosen_model = chosen_model
        self.model_name = self.get_model_name() # store model name as an instance var
        self.model = None # intialize as None. Later represented as HF pipeline. 
        
    def get_model_name(self): 
        '''
        Get full model name from a shorter, chosen name.
        '''

        model_names = {
            "falcon": "tiiuae/falcon-7b",
            "falcon_instruct": "tiiuae/falcon-7b-instruct", 
            "t5":"google/flan-t5-xxl",
            "beluga":"stabilityai/StableBeluga-7B", 
            "llama2": "meta-llama/Llama-2-7b-hf", 
            "llama2_chat":"meta-llama/Llama-2-7b-chat-hf"
        }

        return model_names.get(self.chosen_model)

    def initialize_model(self): 
        '''
        Initialize model if not already done. Common method for pipeline. Overridden by subclasses
        '''
        if self.model is None: 
            self.model = pipeline(model = self.model_name, device_map = "auto")
    
    def completions_generator(self, df, prompt_col, min_len, max_tokens, outfilepath=None):
        '''
        Create completions based on source text in dataframe (df). Save to outfilepath if specified.

        Args
            df: dataframe with prompt col 
            prompt_col: name of column to generate completions from 
            min_len: minimum length of the completion (output)
            max_tokens: maximum new tokens to be added 
            outfilepath: path where the file should be saved (defaults to none, not saving anything)

        Returns
            completions_df: dataframe with model completions and ID 
        '''

        # init model if needed (will only do once)
        self.initialize_model() 
        
        # empty list for completions
        completions = []

        for prompt in tqdm(df[prompt_col], desc="Generating"):
            # use initlaised model to create completion
            completion = self.model(prompt, min_length=min_len, max_new_tokens=max_tokens)

            # extraxt ONLY the text from the completion (it is wrapped as a list of dicts otherwise)
            completion_txt = list(completion[0].values())[0]

            # append to lst
            completions.append(completion_txt)

        # add ID column 
        completions_df = df[["id", prompt_col]].copy()
        
        # add chosen_mdl name to column
        completions_df[f"{self.chosen_model}_completions"] = completions

        # save to outfilepath if not None
        if outfilepath is not None:
            completions_df.to_json(outfilepath, orient="records", lines=True, force_ascii=False)

        return completions_df

class BelugaModel(BaseModel):
    def __init__(self):
        super().__init__(chosen_model="beluga") # only one chosen_model, so it is specified here, inherited by BaseModel. 

    def initialize_model(self):
        if self.model is None: 
            self.model = pipeline(
                model=self.get_model_name(),  # get mdl name from base class
                device_map = "auto",
                return_full_text=False
            )

class Llama2Model(BaseModel):
    def initialize_model(self):
        if self.model is None: 
            self.model = pipeline(
                model=self.get_model_name(),  # get mdl name from base class
                device_map="auto",
                return_full_text=False
            )

class FalconModel(BaseModel):
    def initialize_model(self):
        if self.model is None:
            # init tokenizer for falcon 
            tokenizer = AutoTokenizer.from_pretrained(self.get_model_name())  # get mdl name from base class

            self.model = pipeline(
                model=self.get_model_name(),  # get mdl name from base class
                tokenizer=tokenizer,
                trust_remote_code=True, # trust remote code for falcon
                device_map="auto",
                return_full_text=False, 
            )

def generation_pipeline(chosen_model, df, datafile, prompt_number, min_len, max_tokens, outfilepath=None):
    if chosen_model == "beluga":
        model_instance = BelugaModel()
    
    elif "llama2" in chosen_model:
        model_instance = Llama2Model(chosen_model)

    elif "falcon" in chosen_model: 
        model_instance = FalconModel(chosen_model)

    else:
        model_instance = BaseModel(chosen_model)  # init BaseModel for other models than the specified ones

    # create prompt col 
    if chosen_model == "beluga" or chosen_model == "llama2_chat":
        pg = SpecialPromptGenerator(prompt_number, chosen_model)
        df = pg.format_prompt(df, datafile, chosen_model)
    else: 
        pg = PromptGenerator(prompt_number)
        df = pg.create_prompt(df, datafile)

    # create completions with completions generator from BaseModel
    df_completions = model_instance.completions_generator(df, f"prompt_{prompt_number}", min_len, max_tokens, outfilepath=outfilepath)

    return df_completions