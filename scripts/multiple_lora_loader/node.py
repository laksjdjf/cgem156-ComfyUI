import comfy
import folder_paths
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, NODE_SURFIX, SYMBOL
from .flux_map import FLUX_MAP

CATEGORY_NAME = ROOT_NAME + "multiple_lora_loader"


# Module-level cache replacing the old per-instance `self.loaded_lora` dict.
# execute() is now a classmethod (no `self` to hold state), so the cache is
# keyed by (unique_id, slot_key): unique_id identifies the node instance in the
# graph (via the hidden UNIQUE_ID input) and slot_key identifies the lora slot
# within that node (an int index for the fixed loaders, a slot name for the
# dynamic loader). This reproduces the exact old granularity -- one cache entry
# per lora slot per node instance -- just relocated out of `self`.
_lora_cache = {}


def _load_lora(unique_id, slot_key, model, clip, lora_name, strength_model, strength_clip):
    """Load (with caching + flux key remapping) and apply a single LoRA slot."""
    if strength_model == 0 and strength_clip == 0:
        return model, clip

    lora_path = folder_paths.get_full_path("loras", lora_name)
    cache_key = (unique_id, slot_key)
    cached = _lora_cache.get(cache_key)

    if cached is not None and cached[0] == lora_path:
        new_lora = cached[1]
    else:
        if cached is not None:
            del _lora_cache[cache_key]
        state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        new_lora = {}
        for key, value in state_dict.items():
            new_lora[FLUX_MAP.get(key, key)] = value
        del state_dict
        _lora_cache[cache_key] = (lora_path, new_lora)

    model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, new_lora, strength_model, strength_clip)
    return model_lora, clip_lora


def _multiple_lora_loader(unique_id, model, clip, normalize, normalize_sum, slots):
    """Shared merge logic used by both the fixed-slot and dynamic loaders.

    `slots` is an ordered list of (slot_key, lora_name, strength_model, apply).
    Behavior (including the normalize division) is byte-for-byte the same math
    as the original per-instance implementation; only the cache storage moved.
    """
    lora_names = [s[1] for s in slots]
    strength_models = [s[2] for s in slots]
    applys = [s[3] for s in slots]

    for i, lora_name in enumerate(lora_names):
        if lora_name == "None":
            applys[i] = False

    strength_sum = 0
    for i in range(len(slots)):
        if applys[i]:
            strength_sum += strength_models[i]

    if normalize:
        scale = normalize_sum / strength_sum
    else:
        scale = 1.0

    for i, (slot_key, lora_name, strength_model, apply) in enumerate(slots):
        if not applys[i]:
            continue
        scaled_strength = strength_model * scale
        model, clip = _load_lora(unique_id, slot_key, model, clip, lora_name, scaled_strength, scaled_strength)

    return model, clip


def create_class(num_loras):
    """Build a V3 (io.ComfyNode) class exposing `num_loras` fixed LoRA slots.

    Kept for backward compatibility with existing workflows (config.txt still
    drives how many fixed-size variants get registered). Node ids, input
    names/order and defaults are unchanged from the pre-V3 implementation.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = [
            io.Model.Input("model"),
            io.Boolean.Input("normalize", default=False),
            io.Float.Input("normalize_sum", default=1.0, min=-50.0, max=50.0, step=0.01, round=0.001),
        ]
        lora_options = ["None"] + folder_paths.get_filename_list("loras")
        for i in range(num_loras):
            inputs.append(io.Combo.Input(f"lora_name_{i}", options=lora_options))
            inputs.append(io.Float.Input(f"strength_model_{i}", default=1.0, min=-20.0, max=20.0, step=0.01, round=0.001))
            inputs.append(io.Boolean.Input(f"apply_{i}", default=True))
        inputs.append(io.Clip.Input("clip_optional", optional=True))

        return io.Schema(
            node_id=f"MultipleLoraLoader{num_loras}{NODE_SURFIX}",
            display_name=f"MultipleLoraLoader{num_loras} {SYMBOL}",
            category=CATEGORY_NAME,
            description=f"Fixed {num_loras}-slot multi-LoRA loader. Slot counts are configured in config.txt.",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, normalize, normalize_sum, clip_optional=None, **kwargs) -> io.NodeOutput:
        clip = clip_optional

        slots = [
            (i, kwargs[f"lora_name_{i}"], kwargs[f"strength_model_{i}"], kwargs[f"apply_{i}"])
            for i in range(num_loras)
        ]

        model, clip = _multiple_lora_loader(cls.hidden.unique_id, model, clip, normalize, normalize_sum, slots)
        return io.NodeOutput(model, clip)

    return type(
        f"MultipleLoraLoader{num_loras}",
        (io.ComfyNode,),
        {
            "define_schema": define_schema,
            "execute": execute,
        },
    )

