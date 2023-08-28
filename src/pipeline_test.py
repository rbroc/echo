from transformers import pipeline, AutoTokenizer 
import argparse, pathlib
from modules.pipeline_fns import load_file
from modules.prompt_fns import PromptGenerator, SpecialPromptGenerator
from tqdm import tqdm

class BaseModel():
    '''Basic model for generating text'''

    def __init__(self, chosen_model):
        self.chosen_model = chosen_model
        self.model_name = self.get_model_name() # store model name as an instance var
        self.model = None # intialise as None
        
    def get_model_name(self): 
        '''Get full model name from shorter, chosen name'''

        model_names = {
            "falcon": "tiiuae/falcon-7b",
            "falcon_instruct": "tiiuae/falcon-7b-instruct", 
            "t5":"google/flan-t5-xxl",
            "beluga":"stabilityai/StableBeluga-7B", 
            "llama2": "meta-llama/Llama-2-7b-hf", 
            "llama2_chat":"meta-llama/Llama-2-7b-chat-hf"
        }

        return model_names.get(self.chosen_model)

    def initialize_model(self): # common method for pipeline, overriden by subclasses 
        '''Init model if not already done'''
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

        self.initialize_model()  # init model if needed (will only do once)
        
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
        
        # add chosen_mdl name to 
        completions_df[f"{self.chosen_model}_completions"] = completions

        # save to outfilepath if not None
        if outfilepath is not None:
            completions_df.to_json(outfilepath, orient="records", lines=True, force_ascii=False)

        return completions_df

class BelugaModel(BaseModel):
    def __init__(self):
        super().__init__(chosen_model="beluga") # only one chosen_model, so it is specified here 

    def initialize_model(self):
        self.model = pipeline(
            model=self.get_model_name(),  # get mdl name from base class
            device_map = "auto",
            return_full_text=False
        )

class Llama2Model(BaseModel):
    def initialize_model(self):
        self.model = pipeline(
            model=self.get_model_name(),  # get mdl name from base class
            device_map="auto",
            return_full_text=False
        )

class FalconModel(BaseModel):
    def initialize_model(self):
        # init tokenizer for falcon 
        tokenizer = AutoTokenizer.from_pretrained(self.get_model_name())  # get mdl name from base class

        self.model = pipeline(
            model=self.get_model_name(),  # get mdl name from base class
            tokenizer=tokenizer,
            trust_remote_code=True, # trust remote code for falcon
            device_map="auto",
            return_full_text=False
        )

def extract_min_max_tokens(datafilename):
    if "stories" in datafilename:
        min_tokens = 365
        max_new_tokens = 998
    
    if "mrpc" in datafilename:
        min_tokens = 19
        max_new_tokens = 28
    
    if "dailymail_cnn" in datafilename:
        min_tokens = 44
        max_new_tokens = 70
    
    if "dailydialog" in datafilename: 
        min_tokens = 6
        max_new_tokens = 16

    return min_tokens, max_new_tokens

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

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-f", "--filename", help = "pick which dataset you want", type = str, default = "stories")
    parser.add_argument("-mdl", "--chosen_model", help = "Choose between ...", type = str, default = "t5")
    parser.add_argument("-prompt_n", "--prompt_number", help = "choose which prompt to use", type = int, default = 1)
    parser.add_argument("-subset", "--data_subset", help = "how many rows you want to include. Useful for testing. Defaults to None.", type = int, default=None)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main(): 
    # intialise arguments 
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[1] / "datasets" / args.filename
    datafile = datapath / "data.ndjson"
    #outfile = datapath / f"{args.chosen_model}_data.ndjson"

    # load stuff
    df = load_file(datafile)

    # subset (temporary for testing)
    if args.data_subset is not None: 
        df = df[:args.data_subset]

    # define min and max length 
    min_len, max_tokens = extract_min_max_tokens(args.filename)

    if "llama2" in args.chosen_model:
        from huggingface_hub import login

        # get token from txt
        with open(path.parents[1] / "tokens" / "hf_token.txt") as f:
            hf_token = f.read()

        login(hf_token)

    # run pipeline 
    completions_df = generation_pipeline(
        chosen_model = args.chosen_model, 
        df = df, 
        datafile = args.filename, 
        prompt_number = args.prompt_number, 
        min_len = min_len, 
        max_tokens = max_tokens, 
        outfilepath = path.parents[0]/"test.ndjson")


if __name__ == "__main__":
    main()