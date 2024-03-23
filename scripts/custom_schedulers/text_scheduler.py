'''
load from sampling/custom_sampling/scheulers
input text like "999,893,...,156"
connect to SamplerCustom
'''

import torch
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "custom_schedulers"

class TextScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{"model": ("MODEL",), "timesteps": ("STRING", {"multiline": True}), "verbose": ("BOOLEAN", )}}
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = CATEGORY_NAME

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, timesteps, verbose):
        timesteps = [float(timestep) for timestep in timesteps.replace(" ", "").split(",")]
        sigmas = model.model.model_sampling.sigma(torch.tensor(timesteps))
        sigmas = torch.cat([sigmas, torch.tensor([0])])

        if verbose:
            print("sigmas:", sigmas.tolist())
        return (sigmas, )

NODE_CLASS_MAPPINGS = {
    "TextScheduler": TextScheduler,
}
