import comfy
import folder_paths
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, NODE_SURFIX, SYMBOL
from .flux_map import FLUX_MAP

CATEGORY_NAME = ROOT_NAME + "multiple_lora_loader"

# Generous upper bound for the dynamic (Autogrow) loader below. Autogrow needs a
# fixed ceiling to pre-register optional slots against; growth just reveals more
# of them, it doesn't truly create unlimited inputs.
MAX_DYNAMIC_LORAS = 50

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
            description=(
                f"Fixed {num_loras}-slot multi-LoRA loader. Superseded by "
                f"MultipleLoraLoaderDynamic, which grows to any number of slots; "
                f"kept so existing workflows keep loading."
            ),
            inputs=inputs,
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
            ],
            hidden=[io.Hidden.unique_id],
            is_deprecated=True,
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


class MultipleLoraLoaderDynamic(io.ComfyNode):
    """Truly dynamic multi-LoRA loader built on ComfyUI's V3 Autogrow inputs.

    Design note (Autogrow limitation): Autogrow's per-slot template accepts
    exactly one Input widget -- its own assertion forbids nesting a
    DynamicInput (e.g. DynamicSlot, which is what would be needed to bundle a
    combo + float + bool into a single grow-able "row") as the template. There
    is no first-party way, as of this API version, to grow one row containing
    multiple heterogeneous widgets at once. The best available approximation
    is three parallel Autogrow groups -- lora_names / strength_models / applys
    -- sharing the same "<field>_<index>" naming the old fixed loaders used.
    Growing/removing is therefore per-column rather than per-row: a user could
    in principle grow "lora_name_3" without growing "strength_model_3". To keep
    that harmless, execute() below treats any missing sibling as its sensible
    default (lora_name "None" => slot skipped, strength 1.0, apply True)
    instead of erroring. Users should grow all three columns together to get
    the expected per-row behavior; this is called out for manual/browser-side
    verification since it can't be enforced from the schema alone.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        lora_options = ["None"] + folder_paths.get_filename_list("loras")

        name_template = io.Autogrow.TemplatePrefix(
            input=io.Combo.Input("lora_name", options=lora_options),
            prefix="lora_name_",
            min=0,
            max=MAX_DYNAMIC_LORAS,
        )
        strength_template = io.Autogrow.TemplatePrefix(
            input=io.Float.Input("strength_model", default=1.0, min=-20.0, max=20.0, step=0.01, round=0.001),
            prefix="strength_model_",
            min=0,
            max=MAX_DYNAMIC_LORAS,
        )
        apply_template = io.Autogrow.TemplatePrefix(
            input=io.Boolean.Input("apply", default=True),
            prefix="apply_",
            min=0,
            max=MAX_DYNAMIC_LORAS,
        )

        return io.Schema(
            node_id=f"MultipleLoraLoaderDynamic{NODE_SURFIX}",
            display_name=f"MultipleLoraLoaderDynamic {SYMBOL}",
            category=CATEGORY_NAME,
            description=(
                "Dynamically growable multi-LoRA loader (no fixed slot count). "
                "Grow the lora_names / strength_models / applys inputs together "
                "(same index suffix, e.g. _0, _1, ...) to add a LoRA slot."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("normalize", default=False),
                io.Float.Input("normalize_sum", default=1.0, min=-50.0, max=50.0, step=0.01, round=0.001),
                io.Autogrow.Input("lora_names", template=name_template),
                io.Autogrow.Input("strength_models", template=strength_template),
                io.Autogrow.Input("applys", template=apply_template),
                io.Clip.Input("clip_optional", optional=True),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, normalize, normalize_sum, lora_names, strength_models, applys, clip_optional=None) -> io.NodeOutput:
        clip = clip_optional

        indices = set()
        for grown in (lora_names, strength_models, applys):
            for slot_id in grown:
                indices.add(int(slot_id.rsplit("_", 1)[-1]))

        slots = []
        for i in sorted(indices):
            lora_name = lora_names.get(f"lora_name_{i}", "None")
            strength_model = strength_models.get(f"strength_model_{i}", 1.0)
            apply = applys.get(f"apply_{i}", True)
            slots.append((f"dyn_{i}", lora_name, strength_model, apply))

        model, clip = _multiple_lora_loader(cls.hidden.unique_id, model, clip, normalize, normalize_sum, slots)
        return io.NodeOutput(model, clip)
