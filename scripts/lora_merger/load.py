import comfy
import folder_paths
import os
import re
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, NODE_SURFIX, SYMBOL

CATEGORY_NAME = ROOT_NAME + "lora_merger"

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PRESET_FILE = os.path.join(CURRENT_DIR, "preset.txt")


def extract_numbers(s):
    return [int(num) for num in re.findall(r'\d+', s)]

def expand_lbw(weight_list):
    length = len(weight_list)
    if length == 17:
        new_list = []
        j = 0
        for i in range(26):
            if i in LBW17TO26:
                new_list.append(0.0)
            else:
                new_list.append(weight_list[j])
                j += 1
    elif length == 12:
        new_list = []
        j = 0
        for i in range(20):
            if i in LBW12TO20:
                new_list.append(0.0)
            else:
                new_list.append(weight_list[j])
                j += 1
    else:
        new_list = weight_list
    return new_list

def parse_weight_preset(text):
    lines = text.strip().split("\n")
    weight_dict = {}
    for line in lines:
        key, values = line.split(":")
        float_values = [float(x) for x in values.split(",")]
        weight_dict[key] = float_values
    return weight_dict

def parse_weight_list(text):
    if os.path.exists(PRESET_FILE):
        with open(PRESET_FILE, "r") as f:
            dic = parse_weight_preset(f.read())
    else:
        dic = {}

    if text in dic:
        return dic[text]
    else:
        return [float(weight) for weight in text.split(",")]

LBW17TO26 = [1, 4, 7, 10, 11, 12, 14, 15, 16]
LBW12TO20 = [1, 2, 3, 4, 7, 17, 18, 19]

MID_ID = {26:13, 20:10}

class LoraLoaderFromWeight(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"LoraLoaderFromWeight{NODE_SURFIX}",
            display_name=f"LoRA Loader From Weight {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("LoRA").Input("lora"),
                io.Model.Input("model"),
                io.Clip.Input("clip_optional", optional=True),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, lora, model, clip_optional=None) -> io.NodeOutput:
        lora_weight = lora["lora"]
        strength_model = lora["strength_model"]
        strength_clip = lora["strength_clip"]

        if strength_model == 0 and strength_clip == 0:
            return io.NodeOutput(model, clip_optional)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip_optional, lora_weight, strength_model, strength_clip)
        return io.NodeOutput(model_lora, clip_lora)

# module-level cache replacing the old per-instance `self.loaded_lora` /
# `self.lbw` state (execute() is a classmethod, no `self` to cache on).
_weight_only_cache = {"loaded_lora": None, "lbw": None}

class LoraLoaderWeightOnly(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"LoraLoaderWeightOnly{NODE_SURFIX}",
            display_name=f"LoRA Loader Weight Only {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Float.Input("strength_clip", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.String.Input("lbw", multiline=False, default=""),
            ],
            outputs=[
                io.Custom("LoRA").Output(),
            ],
        )

    @classmethod
    def execute(cls, lora_name, strength_model, strength_clip, lbw) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None

        if _weight_only_cache["loaded_lora"] is not None:
            if _weight_only_cache["loaded_lora"][0] == lora_path:
                lora = _weight_only_cache["loaded_lora"][1]
            else:
                temp = _weight_only_cache["loaded_lora"]
                _weight_only_cache["loaded_lora"] = None
                del temp

        if lora is None or _weight_only_cache["lbw"] != lbw:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            if lbw != "":
                weight_list = parse_weight_list(lbw)
                weight_list = expand_lbw(weight_list)
                print(f"{lora_name} block weight is :{weight_list}")
                length = len(weight_list)

                strength_clip = strength_clip * weight_list[0]

                up_keys = [key for key in lora.keys() if ("lora_up" in key or "lora_B" in key) and not "lora_te" in key]

                for key in up_keys:
                    ids = extract_numbers(key)
                    if "input_blocks" in key:
                        block_id = ids[0]
                    elif "middle_block" in key:
                        block_id = MID_ID[length]
                    elif "output_blocks" in key:
                        block_id = ids[0] + MID_ID[length] + 1
                    elif "down_blocks" in key:
                        block_id = ids[0]*3 + ids[1] + 1
                        if "down_sampler" in key:
                            block_id += 2
                    elif "mid_block" in key:
                        block_id = MID_ID[length]
                    elif "up_blocks" in key:
                        block_id = ids[0]*3 + ids[1] + MID_ID[length] + 1
                        if "up_sampler" in key:
                            block_id += 2
                    else:
                        block_id = 0
                    #print(key, block_id)
                    weight = weight_list[block_id]
                    if weight != 0.0:
                        lora[key] = lora[key] * weight
                    else:
                        if "lora_up" in key:
                            down_key = key.replace("lora_up", "lora_down")
                            alpha_key = key.replace("lora_up.weight", "alpha")
                        else:
                            down_key = key.replace("lora_B", "lora_A")
                            alpha_key = key.replace("lora_B.weight", "alpha")
                        del lora[key]
                        del lora[down_key]
                        if alpha_key in lora:
                            del lora[alpha_key]
            
            _weight_only_cache["loaded_lora"] = (lora_path, lora)
            _weight_only_cache["lbw"] = lbw

        return io.NodeOutput({"lora": lora, "strength_model": strength_model, "strength_clip": strength_clip})
