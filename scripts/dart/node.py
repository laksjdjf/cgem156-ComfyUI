from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
from transformers.generation.logits_process import UnbatchedClassifierFreeGuidanceLogitsProcessor
import comfy
import torch
import re
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "dart"

class LoadDart:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokenizer": ("STRING", {"default": "p1atdev/dart-v1-sft"}),
                "model": ("STRING", {"default": "p1atdev/dart-v1-sft"}),
            }
        }
    RETURN_TYPES = ("DART_TOKENIZER", "DART_MODEL", )
    FUNCTION = "load"

    CATEGORY = CATEGORY_NAME

    def load(self, tokenizer, model):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        return (tokenizer, model, )
    
class DartPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rating": (["general", "sensitive", "questionable", "explicit", "sfw", "nsfw"], ),
                "copyright": ("STRING", {"default": "original"}),
                "character": ("STRING", {"default": ""}),
                "general": ("STRING", {"multiline": True}),
                "long": (["very_short", "short", "long", "very_long"], {"default": "long"}),
            }
        }
    RETURN_TYPES = ("STRING", )
    FUNCTION = "load"

    CATEGORY = CATEGORY_NAME

    def load(self, rating, copyright, character, general,  long):
        prompt = "<|bos|>"
        prompt += f"<rating>rating:{rating}</rating>"
        prompt += f"<copylight>{copyright}</copyright>"
        prompt += f"<character>{character}</character>"
        prompt += "<general>" + f"<|{long}|>"
        prompt += f"{general}"
        prompt += "<|input_end|>"

        return (prompt, )
    
class DartConfig:
    @classmethod
    def INPUT_TYPES(s):
        input_types = {
            "required": {
                "max_new_tokens": (
                    "INT",
                    {"default": 128, "min": 1, "max": 256, "step": 1},
                ),
                "min_new_tokens": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": (
                    "INT",
                    {"default": 20, "min": 1, "max": 500, "step": 1},
                ),
                "num_beams": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
            },
        }

        return input_types

    RETURN_TYPES = ("DART_CONFIG",)
    FUNCTION = "compose"
    CATEGORY = CATEGORY_NAME

    def compose(self, **kwargs):
        kwargs["temperature"] = float(kwargs["temperature"]) # avoid error
        return (kwargs,)
    
class BanTags:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tokenizer": ("DART_TOKENIZER", ),
                "ban_tags": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING", )
    FUNCTION = "generate"

    CATEGORY = CATEGORY_NAME

    def generate(self, tokenizer, ban_tags):
        ban_tags_result = set()
        patterns = [re.compile(ban_tag) for ban_tag in ban_tags.splitlines()]
        for pattern in patterns:
            for tag in tokenizer.vocab:
                if pattern.match(tag):
                    ban_tags_result.add(tag)
        return (", ".join(ban_tags_result), )
    
class DartGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokenizer": ("DART_TOKENIZER", ),
                "model": ("DART_MODEL", ),
                "prompt": ("STRING", {"default": ""}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional":{
                "config": ("DART_CONFIG", ),
                "negative": ("STRING", {"default": None}),
                "ban_tags": ("STRING", {"default": None}),
            }
        }
    RETURN_TYPES = ("BATCH_STRING", "STRING")
    FUNCTION = "generate"

    CATEGORY = CATEGORY_NAME

    def generate(self, tokenizer, model, prompt, batch_size, seed, config=None, negative=None, ban_tags=None):
        if config:
            config = config
        else:
            config = {
                "max_new_tokens": 128,
                "min_new_tokens": 0,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 100,
                "num_beams": 1,
            }
        
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        generation_config = GenerationConfig.from_pretrained("p1atdev/dart-v1-sft", **config) # こんなんでいいの？
        model.to(comfy.model_management.get_torch_device(), dtype=torch.float16).eval()
        inputs = tokenizer([prompt], return_tensors="pt").input_ids.to(comfy.model_management.get_torch_device()).repeat(batch_size, 1)

        if config["cfg_scale"] != 1.0:
            negative_inputs = tokenizer([negative], return_tensors="pt").input_ids.to(comfy.model_management.get_torch_device()).repeat(batch_size, 1)
            loggits_processor = LogitsProcessorList([
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    guidance_scale=config["cfg_scale"],
                    model=model,
                    unconditional_ids=negative_inputs,
                )
            ])
        else:
            loggits_processor = None

        if ban_tags:
            ban_tags_ids = tokenizer([ban_tags]).input_ids
            bad_words_ids = [[token_id] for token_id in ban_tags_ids[0]]
        else:
            bad_words_ids = None

        with torch.no_grad():
            outputs = model.generate(inputs, generation_config=generation_config, bad_words_ids=bad_words_ids, logits_processor=loggits_processor)

        prompts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # delete rating
        prompts = [", ".join(prompt.split(", ")[1:]) for prompt in prompts]

        strings = "" # for checking
        for i, prompt in enumerate(prompts):
            strings += f"Prompt {i + 1}:\n{prompt}\n\n"

        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

        return (prompts, strings)
    
