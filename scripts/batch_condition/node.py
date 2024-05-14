import torch
import numpy as np
import math
import os
from PIL import Image
from ... import ROOT_NAME


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CATEGORY_NAME = ROOT_NAME + "batch_condition"

def lcm(a, b):
    return a * b // math.gcd(a, b)

def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm

class CLIPTextEncodeBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ), 
                "texts":("BATCH_STRING", )
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, clip, texts):
        conds = []
        pooleds = []
        num_tokens = []
        for text in texts:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conds.append(cond)
            pooleds.append(pooled)
            num_tokens.append(cond.shape[1])
        
        # Make number of tokens equal
        # attn(q, k, v) == attn(q, [k]*n, [v]*n)
        lcm = lcm_for_list(num_tokens)
        repeats = [lcm//num for num in num_tokens]
        conds = torch.cat([cond.repeat(1, repeat, 1) for cond, repeat in zip(conds, repeats)])
        pooleds = torch.cat(pooleds)
        return ([[conds, {"pooled_output": pooleds}]], )
    
class StringInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "text": ("STRING", {"multiline": True})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    CATEGORY = CATEGORY_NAME

    def encode(self, text):
        return (text, )
    
class BatchString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("BATCH_STRING",)
    FUNCTION = "encode"

    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        return ([kwargs[f"text{i+1}"] for i in range(len(kwargs))], )
    
class PrefixString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prefix": ("STRING", {"multiline": True}),
                "prompts": ("BATCH_STRING", )
            }
        }
    RETURN_TYPES = ("BATCH_STRING",)
    FUNCTION = "encode"

    CATEGORY = CATEGORY_NAME

    def encode(self, prefix, prompts):
        return ([prefix + prompt for prompt in prompts], )
        
    
class SaveBatchString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("BATCH_STRING", ),
                "folder": ("STRING", {"default": ""}),
                "extension": ("STRING", {"default": "txt"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_NAME

    def save(self, prompts, folder, extension, seed):
        os.makedirs(os.path.join(CURRENT_DIR, folder), exist_ok=True)
        for i, prompt in enumerate(prompts):
            path = os.path.join(CURRENT_DIR, folder, f"{seed:06}_{i:03}.{extension}")
            with open(path, "w") as f:
                f.write(prompt)
        return {}
    
class SaveImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "folder": ("STRING", {"default": ""}),
                "extension": ("STRING", {"default": "png"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_NAME

    def save(self, images, folder, extension, seed):
        os.makedirs(os.path.join(CURRENT_DIR, folder), exist_ok=True)
        for i, image in enumerate(images):
            path = os.path.join(CURRENT_DIR, folder, f"{seed:06}_{i:03}.{extension}")
            Image.fromarray((image.float().cpu() * 255).numpy().astype('uint8')).save(path)
        return {}
    
class SaveLatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT", ),
                "folder": ("STRING", {"default": ""}),
                "extension": (["npy", "npz"], {"default": "npy"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_NAME

    def save(self, latents, folder, extension, seed):
        os.makedirs(os.path.join(CURRENT_DIR, folder), exist_ok=True)
        for i, latent in enumerate(latents["samples"]):
            path = os.path.join(CURRENT_DIR, folder, f"{seed:06}_{i:03}.{extension}")
            if extension == "npy":
                np.save(path, latent.float().cpu().numpy())
            else:
                original_size = (latent.shape[2] * 8, latent.shape[3] * 8)
                crop_ltrb = (0, 0)
                np.savez(
                    path, 
                    latents=latent.float().cpu().numpy(),
                    original_size=np.array(original_size),
                    crop_ltrb=np.array(crop_ltrb),
                )
        return {}
