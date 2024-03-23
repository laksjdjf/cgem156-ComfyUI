import comfy
import math
import torch
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "lora_merger"
CLAMP_QUANTILE = 0.99

class LoraMerge:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_1": ("LoRA",),
                "mode": (["add", "concat", "svd"], ),
                "rank": ("INT", {
                    "default": 16, 
                    "min": 1, #Minimum value
                    "max": 320, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "device": (["cuda", "cpu"], ),
                "dtype": (["float32", "float16", "bfloat16"], ),
            },
            "optional": {
                "lora_2": ("LoRA",),
            }
        }
    RETURN_TYPES = ("LoRA", )
    FUNCTION = "lora_merge"

    CATEGORY = CATEGORY_NAME

    def lora_merge(self, lora_1, lora_2=None, mode=None, rank=None, threshold=None, device=None, dtype=None):
        
        lora = self.merge(lora_1, lora_2, mode, rank, threshold, device, dtype)

        return (lora, )
    
    @torch.no_grad()
    def merge(self, lora_1, lora_2, mode, rank, threshold, device, dtype):
        # lora = up @ down * alpha / rank

        weight = {}
        dtype = torch.float32 if dtype == "float32" else torch.float16 if dtype == "float16" else torch.bfloat16

        if lora_2 is None:
            lora_2 = {"lora":{}, "strength_model":0, "strength_clip":0}

        keys_1 = [key[: key.rfind(".lora_down")] for key in lora_1["lora"].keys() if ".lora_down" in key]
        keys_2 = [key[: key.rfind(".lora_down")] for key in lora_2["lora"].keys() if ".lora_down" in key]
        keys = list(set(keys_1 + keys_2))
        print(f"Merging {len(keys)} modules")
        print(f"{len(keys)-len(keys_1)} modules only in lora_1")
        print(f"{len(keys)-len(keys_2)} modules only in lora_2")
        pber = comfy.utils.ProgressBar(len(keys))

        for key in keys:
            if key not in keys_1:
                up, down, alpha = calc_up_down_alpha(key, lora_2)
                if mode == "svd":
                    up, down = svd_merge(up, down, None, None, rank, threshold, device)
            elif key not in keys_2:
                up, down, alpha = calc_up_down_alpha(key, lora_1)
                if mode == "svd":
                    up, down = svd_merge(up, down, None, None, rank, threshold, device)
            else:
                up_1, down_1, alpha_1 = calc_up_down_alpha(key, lora_1, add=mode!="add")
                up_2, down_2, alpha_2 = calc_up_down_alpha(key, lora_2, add=mode!="add")

                alpha = alpha_1
                
                # Scale to match alpha_1
                up_2 = up_2 * math.sqrt(alpha_2/alpha) 
                down_2 = down_2 * math.sqrt(alpha_2/alpha)

                up_1 = up_1.to(dtype=dtype)
                down_1 = down_1.to(dtype=dtype)
                up_2 = up_2.to(dtype=dtype)
                down_2 = down_2.to(dtype=dtype)

                # linear to conv 1x1 if needed
                if up_1.dim() != up_2.dim():
                    up_2 = up_2.unsqueeze(2).unsqueeze(3) 
                    down_2 = down_2.unsqueeze(2).unsqueeze(3) 

                if mode == "add":
                    up = up_1 + up_2
                    down = down_1 + down_2
                elif mode == "concat":
                    r_1 = up_1.shape[1]
                    r_2 = up_2.shape[1]
                    scale_1 = math.sqrt((r_1+r_2)/r_1)
                    scale_2 = math.sqrt((r_1+r_2)/r_2)
                    up = torch.cat([up_1*scale_1, up_2*scale_2], dim=1)
                    down = torch.cat([down_1*scale_1, down_2*scale_2], dim=0)
                elif mode == "svd":
                    up, down = svd_merge(up_1, down_1, up_2, down_2, rank, threshold, device)
                
            weight[key + ".lora_up.weight"] = up
            weight[key + ".lora_down.weight"] = down
            weight[key + ".alpha"] = alpha

            pber.update(1)
        
        for key, value in lora_1["lora"].items():
            if "lora" not in key:
                weight[key] = value
        
        return {"lora":weight, "strength_model":1, "strength_clip":1}
    
@torch.no_grad()
def calc_up_down_alpha(key, lora, add=True):
    up_key = key + ".lora_up.weight"
    down_key = key + ".lora_down.weight"
    alpha_key = key + ".alpha"

    is_te = "lora_te" in key

    scale = lora["strength_clip"] if is_te else lora["strength_model"]
    sqrt_scale = math.sqrt(abs(scale)) if add else abs(scale) 
    sign_scale = -1 if scale < 0 else 1

    up = lora["lora"][up_key] * sqrt_scale * sign_scale
    down = lora["lora"][down_key] * sqrt_scale
    alpha = lora["lora"][alpha_key]

    return up, down, alpha

# frovenius normによるrankの計算
def index_sv_fro(S, target):
    S_squared = S.pow(2)
    s_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0)/s_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
    index = max(1, min(index, len(S)-1))

    return index

@torch.no_grad()
def svd_merge(up_1, down_1, up_2, down_2, rank, threshold, device=None):
    org_device = up_1.device
    org_dtype = up_1.dtype

    up_1 = up_1.to(device)
    down_1 = down_1.to(device)
    r_1 = up_1.shape[1]
    weight_1 = up_1.view(-1, r_1) @ down_1.view(r_1, -1)

    if up_2 is not None:
        up_2 = up_2.to(device)
        down_2 = down_2.to(device)
        r_2 = up_2.shape[1]
        weight_2 = up_2.view(-1, r_2) @ down_2.view(r_2, -1)
        weight = weight_1 / r_1 + weight_2 / r_2
    else:
        weight = weight_1 / r_1

    weight = weight.to(dtype=torch.float32) # SVD only supports float32

    U, S, Vh = torch.linalg.svd(weight)

    if threshold < 1:
        rank = index_sv_fro(S, threshold) + 1

    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)

    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)

    if down_1.dim() == 4:
        U = U.reshape(up_1.shape[0], rank, 1, 1)
        Vh = Vh.reshape(rank, down_1.shape[1], down_1.shape[2], down_1.shape[3])

    up = U.to(org_device, dtype=org_dtype) * math.sqrt(rank)
    down = Vh.to(org_device, dtype=org_dtype) * math.sqrt(rank)

    return up, down