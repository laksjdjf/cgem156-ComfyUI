import comfy
import math
import torch
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, NODE_SURFIX, SYMBOL

CATEGORY_NAME = ROOT_NAME + "lora_merger"
CLAMP_QUANTILE = 0.99
REGULAR_LORA = "regular"
DIFFUSERS_LORA = "diffusers"

class LoraMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"LoraMerger{NODE_SURFIX}",
            display_name=f"LoRA Merge {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("LoRA").Input("lora_1"),
                io.Combo.Input("mode", options=["add", "concat", "svd", "svd_fast"]),
                io.Int.Input(
                    "rank",
                    default=16,  # Minimum value
                    min=1,
                    max=320,  # Maximum value
                    step=1,  # Slider's step
                    display_mode=io.NumberDisplay.number,  # Cosmetic only: display as "number" or "slider"
                ),
                io.Float.Input(
                    "threshold",
                    default=1.0,
                    min=0,
                    max=1,
                    step=0.01,
                ),
                io.Combo.Input("device", options=["cuda", "cpu"]),
                io.Combo.Input("dtype", options=["float32", "float16", "bfloat16"]),
                io.Custom("LoRA").Input("lora_2", optional=True),
            ],
            outputs=[
                io.Custom("LoRA").Output(),
            ],
        )

    @classmethod
    def execute(cls, lora_1, lora_2=None, mode=None, rank=None, threshold=None, device=None, dtype=None) -> io.NodeOutput:

        lora = cls.merge(lora_1, lora_2, mode, rank, threshold, device, dtype)

        return io.NodeOutput(lora)

    @staticmethod
    @torch.no_grad()
    def merge(lora_1, lora_2, mode, rank, threshold, device, dtype):
        # lora = up @ down * alpha / rank

        weight = {}
        dtype = torch.float32 if dtype == "float32" else torch.float16 if dtype == "float16" else torch.bfloat16

        if lora_2 is None:
            lora_2 = {"lora":{}, "strength_model":0, "strength_clip":0}

        keys_1 = lora_module_keys(lora_1)
        keys_2 = lora_module_keys(lora_2)
        keys = list(set(keys_1 + keys_2))
        print(f"Merging {len(keys)} modules")
        print(f"{len(keys)-len(keys_2)} modules only in lora_1")
        print(f"{len(keys)-len(keys_1)} modules only in lora_2")
        pber = comfy.utils.ProgressBar(len(keys))

        for key in keys:
            output_format = lora_key_format(key, lora_1) or lora_key_format(key, lora_2) or REGULAR_LORA

            if key not in keys_1:
                up, down, alpha = calc_up_down_alpha(key, lora_2)
                if mode in ("svd", "svd_fast"):
                    up, down = svd_merge(up, down, None, None, rank, threshold, device, fast=mode=="svd_fast")
            elif key not in keys_2:
                up, down, alpha = calc_up_down_alpha(key, lora_1)
                if mode in ("svd", "svd_fast"):
                    up, down = svd_merge(up, down, None, None, rank, threshold, device, fast=mode=="svd_fast")
            else:
                up_1, down_1, alpha_1 = calc_up_down_alpha(key, lora_1, add=mode!="add")
                up_2, down_2, alpha_2 = calc_up_down_alpha(key, lora_2, add=mode!="add")

                alpha = alpha_1
                alpha_1_value = alpha_to_float(alpha_1)
                alpha_2_value = alpha_to_float(alpha_2)
                
                # Scale to match alpha_1
                up_2 = up_2 * math.sqrt(alpha_2_value/alpha_1_value) 
                down_2 = down_2 * math.sqrt(alpha_2_value/alpha_1_value)

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
                elif mode in ("svd", "svd_fast"):
                    up, down = svd_merge(up_1, down_1, up_2, down_2, rank, threshold, device, fast=mode=="svd_fast")
                
            set_up_down_alpha(weight, key, up, down, alpha, output_format)

            pber.update(1)
        
        for key, value in lora_1["lora"].items():
            if "lora" not in key:
                weight[key] = value
        
        return {"lora":weight, "strength_model":1, "strength_clip":1}
    
class LoraSVDRank(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"LoraSVDRank{NODE_SURFIX}",
            display_name=f"LoRA SVD Rank {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("LoRA").Input("lora"),
                io.Float.Input(
                    "threshold",
                    default=1.0,
                    min=0,
                    max=1,
                    step=0.001,
                ),
                io.Combo.Input("device", options=["cuda", "cpu"]),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, lora, threshold, device) -> io.NodeOutput:

        keys = lora_module_keys(lora)
        pber = comfy.utils.ProgressBar(len(keys))

        content = ""
        for key in keys:
            up, down, alpha = calc_up_down_alpha(key, lora)
            index = svd_show(up, down, threshold, device)
            content += f"{key}: {index}\n"
            pber.update(1)

        return io.NodeOutput(content)
    
@torch.no_grad()
def calc_up_down_alpha(key, lora, add=True):
    lora_format = lora_key_format(key, lora)
    if lora_format == DIFFUSERS_LORA:
        up_key = key + ".lora_B.weight"
        down_key = key + ".lora_A.weight"
    else:
        up_key = key + ".lora_up.weight"
        down_key = key + ".lora_down.weight"
    alpha_key = key + ".alpha"

    is_te = "lora_te" in key

    scale = lora["strength_clip"] if is_te else lora["strength_model"]
    sqrt_scale = math.sqrt(abs(scale)) if add else abs(scale) 
    sign_scale = -1 if scale < 0 else 1

    up = lora["lora"][up_key] * sqrt_scale * sign_scale
    down = lora["lora"][down_key] * sqrt_scale
    alpha = lora["lora"].get(alpha_key)
    if alpha is None:
        alpha = torch.tensor(down.shape[0], dtype=torch.float32, device=down.device)

    return up, down, alpha

def lora_module_keys(lora):
    keys = set()
    for key in lora["lora"].keys():
        if key.endswith(".lora_down.weight"):
            keys.add(key[: key.rfind(".lora_down.weight")])
        elif key.endswith(".lora_A.weight"):
            keys.add(key[: key.rfind(".lora_A.weight")])
    return list(keys)

def lora_key_format(key, lora):
    state_dict = lora["lora"]
    if key + ".lora_up.weight" in state_dict and key + ".lora_down.weight" in state_dict:
        return REGULAR_LORA
    if key + ".lora_B.weight" in state_dict and key + ".lora_A.weight" in state_dict:
        return DIFFUSERS_LORA
    return None

def set_up_down_alpha(weight, key, up, down, alpha, lora_format):
    if lora_format == DIFFUSERS_LORA:
        weight[key + ".lora_B.weight"] = up
        weight[key + ".lora_A.weight"] = down
    else:
        weight[key + ".lora_up.weight"] = up
        weight[key + ".lora_down.weight"] = down
    weight[key + ".alpha"] = alpha

def alpha_to_float(alpha):
    if torch.is_tensor(alpha):
        return float(alpha.detach().cpu())
    return float(alpha)

# frovenius normによるrankの計算
def index_sv_fro(S, target, total_sq=None):
    S_squared = S.pow(2)
    s_fro_sq = float(torch.sum(S_squared) if total_sq is None else total_sq)
    sum_S_squared = torch.cumsum(S_squared, dim=0)/s_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
    index = max(1, min(index, len(S)-1))

    return index

@torch.no_grad()
def svd_merge(up_1, down_1, up_2, down_2, rank, threshold, device=None, fast=False):
    org_device = up_1.device
    org_dtype = up_1.dtype

    if up_2 is None and threshold >= 1 and rank == up_1.shape[1]:
        return up_1.contiguous(), down_1.contiguous()

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

    total_sq = torch.sum(weight.pow(2)) if fast and threshold < 1 else None

    if fast:
        U, S, Vh = svd_lowrank(weight, rank, threshold, total_sq=total_sq)
    else:
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    if threshold < 1:
        rank = index_sv_fro(S, threshold, total_sq=total_sq) + 1

    rank = min(rank, len(S))
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

    up = (U.to(org_device, dtype=org_dtype) * math.sqrt(rank)).contiguous()
    down = (Vh.to(org_device, dtype=org_dtype) * math.sqrt(rank)).contiguous()

    return up, down

@torch.no_grad()
def svd_lowrank(weight, rank, threshold, total_sq=None, oversample=8, niter=2):
    max_rank = min(weight.shape)
    q = min(max_rank, max(1, rank + oversample))

    if threshold < 1:
        q = min(max_rank, max(q, 32))

    while True:
        U, S, V = torch.svd_lowrank(weight, q=q, niter=niter)
        order = torch.argsort(S, descending=True)
        U = U[:, order]
        S = S[order]
        V = V[:, order]

        if threshold >= 1 or q >= max_rank:
            break

        estimated_rank = index_sv_fro(S, threshold, total_sq=total_sq) + 1
        if estimated_rank < len(S) - 1:
            break

        next_q = min(max_rank, q * 2)
        if next_q == q:
            break
        q = next_q

    return U, S, V.T

@torch.no_grad()
def svd_show(up, down, threshold, device):
    up = up.to(device)
    down = down.to(device)
    rank = up.shape[1]
    weight = up.view(-1, rank) @ down.view(rank, -1)
    weight = weight.to(dtype=torch.float32) # SVD only supports float32

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    if threshold < 1:
        index = index_sv_fro(S, threshold)
    else:
        index = rank

    return index
