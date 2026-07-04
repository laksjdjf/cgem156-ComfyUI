import comfy
import folder_paths
import math
import os
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, NODE_SURFIX, SYMBOL

CATEGORY_NAME = ROOT_NAME + "lora_merger"

class LoraSave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"LoraSave{NODE_SURFIX}",
            display_name=f"LoRA Save {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("LoRA").Input("lora"),
                io.String.Input("file_name", multiline=False, default="merged"),
                io.Combo.Input("extension", options=["safetensors"]),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora, file_name, extension) -> io.NodeOutput:
        save_path = os.path.join(folder_paths.folder_names_and_paths["loras"][0][0], file_name + "." + extension)
        
        if lora["strength_model"] == 1 and lora["strength_clip"] == 1:
            new_state_dict = make_contiguous(lora["lora"])
        else:
            new_state_dict = {}
            for key in lora["lora"].keys():
                scale = lora["strength_clip"] if "lora_te" in key else lora["strength_model"]
                sqrt_scale = math.sqrt(abs(scale))
                sign_scale = 1 if scale >= 0 else -1
                if "lora_up" in key or "lora_B" in key:
                    new_state_dict[key] = lora["lora"][key] * sqrt_scale * sign_scale
                elif "lora_down" in key or "lora_A" in key:
                    new_state_dict[key] = lora["lora"][key] * sqrt_scale
                else:
                    new_state_dict[key] = lora["lora"][key]
            new_state_dict = make_contiguous(new_state_dict)
        print(f"Saving LoRA to {save_path}")
        comfy.utils.save_torch_file(new_state_dict, save_path)

        return io.NodeOutput()

def make_contiguous(state_dict):
    return {key: value.contiguous() if hasattr(value, "contiguous") else value for key, value in state_dict.items()}
