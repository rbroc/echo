'''
Script for loading model pipelines
'''
from abc import ABC, abstractmethod
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM

class Model(): 
    def __init__(self, chosen_model_name):
        self.chosen_model_name = chosen_model_name
        self.full_model_name = self.get_model_name()
        self.model = None
        self.system_prompt = self.get_system_prompt()

    @abstractmethod
    def initialize_model(self):
        '''Intialize model using pipeline(). Subclasses use this differently'''
        pass

    def get_model_name(self): 
        '''
        Get full model name from a shorter, chosen name.
        '''

        model_names = {
            "beluga7b":"stabilityai/StableBeluga-7B", 
            "beluga70b":"stabilityai/StableBeluga2",
            "llama2_chat13b":"meta-llama/Llama-2-13b-chat-hf",
            "beluga70bQ":"TheBloke/StableBeluga2-70B-GPTQ", # GPTQ optimized for GPU
            "llama2_chat70bQ":"TheBloke/Llama-2-70B-Chat-GPTQ"
        }

        if self.chosen_model_name in model_names:
            return model_names[self.chosen_model_name]
        else:
            all_names = ", ".join(model_names.keys())
            raise ValueError(
                f"Passed model is not supported, "
                f"please choose one of {all_names}"
            )

    def get_system_prompt(self):
        '''
        StableBeluga follows this prompt structure: https://huggingface.co/stabilityai/StableBeluga-7B) 
        Llama2 chat versions follow this prompt structure: https://gpus.llm-utils.org/llama-2-prompt-template/ (note that the DEFAULT system prompt is recommended to be removed https://github.com/facebookresearch/llama/commit/a971c41bde81d74f98bc2c2c451da235f1f1d37c. Custom system prompts may be useful. Regardless of whether a system prompt is used, the format below is required to produce intelligble text) 
        '''
        system_prompts = {
            "beluga": "You are StableBeluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n",
            "llama2_chat": "You are an AI, but you do not deviate from the task prompt and you do not small talk. Never begin your response with 'Sure, here is my response: ' or anything of the like. It is important that you finish without getting cut off."
        }

        if "beluga" in self.chosen_model_name:
            return system_prompts.get("beluga", "")

        elif "llama2_chat" in self.chosen_model_name:
            return system_prompts.get("llama2_chat", "")

        else:
            # return empty string for models with no specific system prompt
            return ""

    def format_prompt(self, user_input):
        '''
        Format final prompt with user_input

        Args
            user_input: should include task prompt and source text e.g., ""summarize this: 'I love language models'"

        Returns 
            Formatted prompt 
        '''
        if "beluga" in self.chosen_model_name:
            return f"### System:\n{self.system_prompt}### User: {user_input}\n\n### Assistant:\n"

        elif "llama2_chat" in self.chosen_model_name:
            return f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{user_input} [/INST]"
        
        else:
            return user_input

class FullModel(Model):
    '''
    Full, unquantized models (Beluga and Llama2)
    '''
    def initialize_model(self, cache_dir=None):
        '''
        Init model and tokenizer.
            cache_dir: if cache_dir is specified, downloads model to cache_dir (or loads it if already downloaded). In case of any bugs, delete local folder.
        '''
        if self.model is None:
            model = AutoModelForCausalLM.from_pretrained(self.full_model_name, cache_dir=cache_dir, device_map="auto")

            tokenizer = AutoTokenizer.from_pretrained(self.full_model_name, cache_dir=cache_dir)

            self.model = pipeline(
                    model=model,  # get mdl name from base class
                    torch_dtype=torch.bfloat16,
                    tokenizer=tokenizer,
                    return_full_text=False,
                    task="text-generation"
                )
                
            # allow for padding 
            self.model.tokenizer.pad_token_id = self.model.model.config.eos_token_id

class QuantizedModel(Model):
    '''
    Quantized GPQT models e.g., https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ (optimised for GPU).
    '''
    def initialize_model(self, cache_dir=None):
        '''
        Init model and tokenizer.
            cache_dir: if cache_dir is specified, downloads model to cache_dir (or loads it if already downloaded). In case of any bugs, delete local folder.
        '''
        if self.model is None: 
            model = AutoModelForCausalLM.from_pretrained(self.full_model_name,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main",
                                                cache_dir=cache_dir, 
                                                torch_dtype=torch.float16, 
                                                low_cpu_mem_usage=True
                                                )
                
            tokenizer = AutoTokenizer.from_pretrained(self.full_model_name, use_fast=True, cache_dir=cache_dir)
        
            self.model = pipeline(
                    model=model,
                    device_map="auto", # needs to be here and not in the model arg for quantized mdl 
                    tokenizer=tokenizer,
                    return_full_text=False,
                    task="text-generation"
                )

            # allow for padding 
            self.model.tokenizer.pad_token_id = self.model.model.config.eos_token_id

class vLLM_Model(Model):
    '''
    vLLM model https://github.com/vllm-project/vllm

    Wraps a HF model 
    '''
    def initialize_model(self, cache_dir=None, seed=129):
        if self.model is None:
            # get available gpus
            available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

            self.model = LLM(self.full_model_name, download_dir=cache_dir, tensor_parallel_size=available_gpus, seed=seed, enforce_eager=True)