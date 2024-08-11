import comfy
import folder_paths
from ... import ROOT_NAME
from .flux_map import FLUX_MAP

CATEGORY_NAME = ROOT_NAME + "multiple_lora_loader"

def create_class(num_loras):
    class MultipleLoraLoader:
        def __init__(self):
            self.loaded_lora = {k: None for k in range(num_loras)}

        @classmethod
        def INPUT_TYPES(s):
            required = {"model": ("MODEL", )}

            required["normalize"] = ("BOOLEAN", {"default": False})
            required["normalize_sum"] = ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.01})

            for i in range(num_loras):
                required[f"lora_name_{i}"] = (["None"] + folder_paths.get_filename_list("loras"), )
                required[f"strength_model_{i}"] = ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01})
                required[f"apply_{i}"] = ("BOOLEAN", {"default": True})

            return {"required": required, "optional": {"clip_optional": ("CLIP", )}}
        
        RETURN_TYPES = ("MODEL", "CLIP")
        FUNCTION = "multiple_lora_loader"
        CATEGORY = CATEGORY_NAME

        def multiple_lora_loader(self, **kwargs):

            model = kwargs.get("model")
            clip = kwargs.get("clip_optional", None)

            normalize = kwargs.get("normalize")
            normalize_sum = kwargs.get("normalize_sum")

            lora_names = [kwargs.get(f"lora_name_{i}") for i in range(num_loras)]
            strength_models = [kwargs.get(f"strength_model_{i}") for i in range(num_loras)]
            applys = [kwargs.get(f"apply_{i}") for i in range(num_loras)]

            strength_sum = 0
            for i in range(num_loras):
                if lora_names[i] == "None":
                    applys[i] = False
                
                if applys[i]:
                    strength_sum += strength_models[i]

            if normalize:
                scale = normalize_sum / strength_sum
            else:
                scale = 1.0
            
            for i in range(num_loras):
                lora_name = lora_names[i]
                strength_model = strength_models[i] * scale
                apply = applys[i]

                #print(lora_name, strength_model, apply)

                if apply:
                    model, clip = self.load_lora(model, clip, lora_name, strength_model, strength_model, i)

            return (model, clip)
        
        def load_lora(self, model, clip, lora_name, strength_model, strength_clip, index):
            if strength_model == 0 and strength_clip == 0:
                return (model, clip)

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = None
            if self.loaded_lora[index] is not None:
                if self.loaded_lora[index][0] == lora_path:
                    lora = self.loaded_lora[index][1]
                else:
                    temp = self.loaded_lora[index]
                    self.loaded_lora[index] = None
                    del temp

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                new_lora = {}
                for key, value in lora.items():
                    new_lora[FLUX_MAP.get(key, key)] = value
                del lora

                self.loaded_lora[index] = (lora_path, new_lora)

            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, new_lora, strength_model, strength_clip)
            return (model_lora, clip_lora)
        
    return MultipleLoraLoader