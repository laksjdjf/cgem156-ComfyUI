# ref: ScaleCrafter https://github.com/YingqingHe/ScaleCrafter

import math
import comfy.ops
import torch.nn.functional as F
from comfy_api.v0_0_2 import io
ops = comfy.ops.disable_weight_init

from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "scale-crafter"

class ScaleCrafter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ScaleCrafter|cgem156",
            display_name="Scale Crafter 🍌",
            category=CATEGORY_NAME,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("dilation_rate", default=1, min=0.01, max=10, step=0.01),
                io.Int.Input("depth", default=0, min=0, max=12, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("start", default=0, min=0, max=1000, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("end", default=500, min=0, max=1000, step=1, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, dilation_rate, depth, start, end) -> io.NodeOutput:
        new_model = model.clone()
        org_forwards = {}

        target_dilation = (math.ceil(dilation_rate), math.ceil(dilation_rate))
        target_padding = target_dilation
        interp_rate = target_dilation[0] / dilation_rate

        def forward_hooker(module, forward):
            def forward_hook(x):
                org_size = x.shape[2:]
                module.dilation = target_dilation
                module.padding = target_padding
                if interp_rate != 1.0:
                    x = F.interpolate(x, scale_factor=interp_rate, mode='bicubic', align_corners=False)
                x = forward(x)
                if interp_rate != 1.0:
                    x = F.interpolate(x, size=org_size, mode='bicubic', align_corners=False)
                module.dilation = (1, 1)
                module.padding = (1, 1)
                return x
            return forward_hook

        def replace_conv2d(model):
            for name, module in model.model.diffusion_model.named_modules():
                if isinstance(module, ops.Conv2d) and module.kernel_size == (3, 3) and module.stride == (1, 1) and module.padding == (1, 1):
                    if name.split(".")[0] == "input_blocks":
                        cur_depth = int(name.split(".")[1])
                        max_depth = cur_depth
                    elif name.split(".")[0] == "middle_block":
                        cur_depth = max_depth + 1
                    elif name.split(".")[0] == "output_blocks":
                        cur_depth = max_depth - int(name.split(".")[1])
                    else:
                        cur_depth = 0

                    if cur_depth >= depth:
                        org_forwards[name] = module.forward
                        module.forward = forward_hooker(module, org_forwards[name])

        def restore_conv2d(model):
            for name, module in model.model.diffusion_model.named_modules():
                if name in org_forwards:
                    module.forward = org_forwards[name]
            org_forwards.clear()

        # unet計算前後のパッチ
        def apply_dilate(model_function, kwargs):
            sigmas = kwargs["timestep"]
            t = new_model.model.model_sampling.timestep(sigmas)
            if t[0] < (1000 - end) or t[0] > (1000 - start):
                return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

            replace_conv2d(new_model)
            retval = model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])
            restore_conv2d(new_model)
            return retval

        new_model.set_model_unet_function_wrapper(apply_dilate)

        return io.NodeOutput(new_model)
