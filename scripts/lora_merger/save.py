import comfy
import folder_paths
import math
import os
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "lora_merger"

class LoraSave:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "lora": ("LoRA",),
                              "file_name": ("STRING", {"multiline": False, "default": "merged"}),
                              "extension": (["safetensors"], ),
                              }}
    RETURN_TYPES = ()
    FUNCTION = "lora_save"

    CATEGORY = CATEGORY_NAME

    OUTPUT_NODE = True

    def lora_save(self, lora, file_name, extension):
        save_path = os.path.join(folder_paths.folder_names_and_paths["loras"][0][0], file_name + "." + extension)
        
        if lora["strength_model"] == 1 and lora["strength_clip"] == 1:
            new_state_dict = lora["lora"]  
        else:
            new_state_dict = {}
            for key in lora["lora"].keys():
                scale = lora["strength_clip"] if "lora_te" in key else lora["strength_model"]
                sqrt_scale = math.sqrt(abs(scale))
                sign_scale = 1 if scale >= 0 else -1
                if "lora_up" in key:
                    new_state_dict[key] = lora["lora"][key] * sqrt_scale * sign_scale
                elif "lora_down" in key:
                    new_state_dict[key] = lora["lora"][key] * sqrt_scale
                else:
                    new_state_dict[key] = lora["lora"][key]
        print(f"Saving LoRA to {save_path}")
        comfy.utils.save_torch_file(new_state_dict, save_path)

        return {}