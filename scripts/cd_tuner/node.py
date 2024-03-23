import torch
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "cd-tuner"

class CDTuner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "detail_1": ("FLOAT", {
                    "default": 0,
                    "min": -10,
                    "max": 10,
                    "step": 0.1
                }),
                "detail_2": ("FLOAT", {
                    "default": 0,
                    "min": -10,
                    "max": 10,
                    "step": 0.1
                }),
                "contrast_1": ("FLOAT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 0.1
                }),
                "start": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "end": ("INT", {
                    "default": 1000, 
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "apply"
    CATEGORY = CATEGORY_NAME

    def apply(self, model, detail_1, detail_2, contrast_1, start, end):
        '''
        detail_1: 最初のConv層のweightを減らしbiasを増やすことで、detailを増やす・・？
        detail_2: 最後のConv層前のGroupNormの以下略
        contrast_1: 最後のConv層のbiasの0チャンネル目を増やすことでコントラストを増やす・・・？
        '''
        new_model = model.clone()
        ratios = fineman([detail_1, detail_2, contrast_1])
        self.storedweights = {}
        self.start = start
        self.end = end

        # unet計算前後のパッチ
        def apply_cdtuner(model_function, kwargs):
            t = new_model.model.model_sampling.timestep(kwargs["timestep"])
            if t[0] < (1000 - self.end) or t[0] > (1000 - self.start):
                return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])
            for i, name in enumerate(ADJUSTS):
                # 元の重みをロード
                self.storedweights[name] = getset_nested_module_tensor(True, new_model, name).clone()
                if 4 > i:
                    new_weight = self.storedweights[name] * ratios[i]
                else:
                    device = self.storedweights[name].device
                    dtype = self.storedweights[name].dtype
                    new_weight = self.storedweights[name] + torch.tensor(ratios[i], device=device, dtype=dtype)
                # 重みを書き換え
                getset_nested_module_tensor(False, new_model, name, new_tensor=new_weight)
            retval = model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

            # 重みを元に戻す
            for name in ADJUSTS:
                getset_nested_module_tensor(False, new_model, name, new_tensor=self.storedweights[name])

            return retval

        new_model.set_model_unet_function_wrapper(apply_cdtuner)

        return (new_model, )


def getset_nested_module_tensor(clone, model, tensor_path, new_tensor=None):
    sdmodules = tensor_path.split('.')
    target_module = model
    last_attr = None

    for module_name in sdmodules if clone else sdmodules[:-1]:
        if module_name.isdigit():
            target_module = target_module[int(module_name)]
        else:
            target_module = getattr(target_module, module_name)

    if clone:
        return target_module

    last_attr = sdmodules[-1]
    setattr(target_module, last_attr, torch.nn.Parameter(new_tensor))

# なんでfineman?
def fineman(fine):
    fine = [
        1 - fine[0] * 0.01,
        1 + fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1 + fine[1] * 0.02,
        [fine[2] * 0.02, 0, 0, 0]
    ]
    return fine


ADJUSTS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.input_blocks.0.0.bias",
    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",
    "model.diffusion_model.out.2.bias",
]

