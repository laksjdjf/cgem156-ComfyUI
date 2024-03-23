# ref: ScaleCrafter https://github.com/YingqingHe/ScaleCrafter

import math
import comfy.ops
import torch.nn.functional as F
ops = comfy.ops.disable_weight_init

from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "scale-crafter"

class ScaleCrafter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "dilation_rate": ("FLOAT", {"default": 1, "min": 0.01, "max": 10, "step": 0.01 }),
                "depth": ("INT", {"default": 0, "min": 0, "max": 12, "step": 1, "display": "number"}),
                "start": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "display": "number"}),
                "end": ("INT", {"default": 500, "min": 0, "max": 1000, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "apply"
    CATEGORY = CATEGORY_NAME

    def apply(self, model, dilation_rate, depth, start, end):
        new_model = model.clone()
        self.org_forwards = {}
        self.start = start
        self.end = end
        self.dilation_rate = dilation_rate
        self.depth = depth

        self.target_dilation = (math.ceil(self.dilation_rate), math.ceil(self.dilation_rate))
        self.target_padding = self.target_dilation
        self.interp_rate = self.target_dilation[0] / self.dilation_rate

        # unet計算前後のパッチ
        def apply_dilate(model_function, kwargs):
            sigmas = kwargs["timestep"]
            t = new_model.model.model_sampling.timestep(sigmas)
            if t[0] < (1000 - end) or t[0] > (1000 - start):
                return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])
            
            self.replace_conv2d(new_model)
            retval = model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])
            self.restore_conv2d(new_model)
            return retval

        new_model.set_model_unet_function_wrapper(apply_dilate)

        return (new_model, )
    
    def replace_conv2d(self, model):
        for name, module in model.model.diffusion_model.named_modules():
            if isinstance(module, ops.Conv2d) and module.kernel_size == (3, 3) and module.stride == (1, 1) and module.padding == (1, 1):
                if name.split(".")[0] == "input_blocks":
                    depth = int(name.split(".")[1])
                    max_depth = depth
                elif name.split(".")[0] == "middle_block":
                    depth = max_depth + 1
                elif name.split(".")[0] == "output_blocks":
                    depth = max_depth - int(name.split(".")[1])
                else:
                    depth = 0

                if depth >= self.depth:
                    self.org_forwards[name] = module.forward
                    module.forward = self.forward_hooker(module, self.org_forwards[name])
    
    def restore_conv2d(self, model):
        for name, module in model.model.diffusion_model.named_modules():
            if name in self.org_forwards:
                module.forward = self.org_forwards[name]
        self.org_forwards = {}

    def forward_hooker(self, module, forward):
        def forward_hook(x):
            org_size = x.shape[2:]
            module.dilation = self.target_dilation
            module.padding = self.target_padding
            if self.interp_rate != 1.0:
                x = F.interpolate(x, scale_factor=self.interp_rate, mode='bicubic', align_corners=False)
            x = forward(x)
            if self.interp_rate != 1.0:
                x = F.interpolate(x, size=org_size, mode='bicubic', align_corners=False)
            module.dilation = (1, 1)
            module.padding = (1, 1)
            return x
        return forward_hook

