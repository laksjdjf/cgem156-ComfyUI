import comfy
import folder_paths
from .input_hint import ControlNetConditioningEmbedding
import torch.nn.functional as F
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "lortnoc"

class LortnocLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "image": ("IMAGE", ),
                              "file_name": (folder_paths.get_filename_list("controlnet"), ),
                              "strength_lora": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_hint": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "load_lortnoc"

    CATEGORY = CATEGORY_NAME

    def load_lortnoc(self, model, image, file_name, strength_lora, strength_hint):
        if strength_lora == 0 and strength_hint == 0:
            return (model, )

        lora_path = folder_paths.get_full_path("controlnet", file_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
            lora = {k:v for k, v in state_dict.items() if "lora" in k}
            self.input_hint_sd = {".".join(k.split(".")[1:]):v for k, v in state_dict.items() if "lora" not in k}
            self.loaded_lora = (lora_path, lora)
        
            self.input_hint = ControlNetConditioningEmbedding(320, 3)
            self.input_hint.load_state_dict(self.input_hint_sd)

        self.hint = self.input_hint(image.permute(0, 3, 1, 2))

        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_lora, None)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == 0:
                size = h.shape[2:]
                if size != self.hint.shape[2:]:
                    hint = F.interpolate(self.hint, size, mode="bilinear", align_corners=False).to(h)
                else:
                    hint = self.hint.to(h)
                h = h + hint * strength_hint
            
            return h
            
        model_lora.set_model_input_block_patch(input_block_patch)
        return (model_lora, )