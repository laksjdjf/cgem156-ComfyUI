from comfy.samplers import KSAMPLER
import torch
from comfy.k_diffusion.sampling import default_noise_sampler, to_d
from tqdm.auto import trange

from ... import ROOT_NAME

@torch.no_grad()
def sampler_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d = to_d(x, sigmas[i], denoised)

        sigma_from = sigmas[i]
        sigma_to = sigmas[i + 1]

        t = model.inner_model.inner_model.model_sampling.timestep(sigma_from)
        down_t = (1 - gamma) * t
        sigma_down = model.inner_model.inner_model.model_sampling.sigma(down_t)

        if sigma_down > sigma_to:
            sigma_down = sigma_to

        sigma_up = (sigma_to ** 2 - sigma_down ** 2) ** 0.5
        
        # same as euler ancestral
        d = to_d(x, sigma_from, denoised)
        dt = sigma_down - sigma_from
        x = x + d * dt
        if sigma_to > 0:
            x = x + noise_sampler(sigma_from, sigma_to) * sigma_up
    return x

class TCDSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "gamma": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step":0.01}),
            },
        }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = ROOT_NAME + "custom_samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, gamma):
        sampler = KSAMPLER(sampler_tcd, {"gamma": gamma})
        return (sampler, )
    
NODE_CLASS_MAPPINGS = {
    "TCDSampler": TCDSampler,
}