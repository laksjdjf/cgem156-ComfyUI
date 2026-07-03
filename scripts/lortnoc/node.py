import comfy
import folder_paths
from .input_hint import ControlNetConditioningEmbedding
import torch.nn.functional as F
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "lortnoc"

# module-level cache replacing the old per-instance `self.loaded_lora` /
# `self.input_hint` state (execute() is a classmethod, no `self` to cache on).
_lortnoc_cache = {"loaded_lora": None, "input_hint": None}

class LortnocLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LortnocLoader|cgem156",
            display_name="Lortnoc Loader 🍌",
            category=CATEGORY_NAME,
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("image"),
                io.Combo.Input("file_name", options=folder_paths.get_filename_list("controlnet")),
                io.Float.Input("strength_lora", default=1.0, min=-20.0, max=20.0, step=0.01),
                io.Float.Input("strength_hint", default=1.0, min=-20.0, max=20.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, image, file_name, strength_lora, strength_hint) -> io.NodeOutput:
        if strength_lora == 0 and strength_hint == 0:
            return io.NodeOutput(model)

        lora_path = folder_paths.get_full_path("controlnet", file_name)
        lora = None
        loaded_lora = _lortnoc_cache["loaded_lora"]
        if loaded_lora is not None:
            if loaded_lora[0] == lora_path:
                lora = loaded_lora[1]
            else:
                _lortnoc_cache["loaded_lora"] = None

        if lora is None:
            state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
            lora = {k:v for k, v in state_dict.items() if "lora" in k}
            input_hint_sd = {".".join(k.split(".")[1:]):v for k, v in state_dict.items() if "lora" not in k}
            _lortnoc_cache["loaded_lora"] = (lora_path, lora)

            input_hint = ControlNetConditioningEmbedding(320, 3)
            input_hint.load_state_dict(input_hint_sd)
            _lortnoc_cache["input_hint"] = input_hint

        hint = _lortnoc_cache["input_hint"](image.permute(0, 3, 1, 2))

        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_lora, None)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == 0:
                size = h.shape[2:]
                if size != hint.shape[2:]:
                    hint_resized = F.interpolate(hint, size, mode="bilinear", align_corners=False).to(h)
                else:
                    hint_resized = hint.to(h)
                h = h + hint_resized * strength_hint

            return h

        model_lora.set_model_input_block_patch(input_block_patch)
        return io.NodeOutput(model_lora)
