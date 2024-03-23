'''
    Implementation of RCFG in https://arxiv.org/abs/2312.12491
    Node is in sampling/custom_sampling/samplers
    original_latent is OPTIONAL
    If original_latent is set, it is Self-Negative else Onetime-Negative
    cfg is recommendet near 1.0 (KSAMPLER"s cfg is ignored)
    delta is よくわかんない
'''

from comfy.samplers import KSAMPLER
import torch
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange
import copy

from ... import ROOT_NAME

@torch.no_grad()
def sampler_lcm_rcfg(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, enable=True, delta=1.0, cfg=1.0, original_latent=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x)
    s_in = x.new_ones([x.shape[0]])
    
    if enable:
        extra_args["cond_scale"] = 1.0
        
        uncond = extra_args["uncond"]
        extra_args_uncond = copy.copy(extra_args)

        extra_args_uncond["cond"] = uncond
    
    if original_latent is None or not enable:
        denoised_uncond = None
    else:
        denoised_uncond = model.inner_model.inner_model.process_latent_in(original_latent).to(x)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        noise_pred = (x - denoised) / sigmas[i]

        if enable and denoised_uncond is None:
            denoised_uncond = model(x, sigmas[i] * s_in, **extra_args_uncond)
        
        if denoised_uncond is not None:
            noise_pred_uncond = (x - denoised_uncond) / sigmas[i] * delta
            noise_pred_cfg = noise_pred_uncond + cfg * (noise_pred - noise_pred_uncond)
            denoised = x - noise_pred_cfg * sigmas[i]

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
            
    return x

class LCMSamplerRCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "enable": ("BOOLEAN", {"default": True}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step":0.01, "round": False}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step":0.01, "round": False}),
            },
            "optional":{
                "original_latent": ("LATENT",),
            }
        }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = ROOT_NAME + "custom_samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, enable, delta, cfg, original_latent=None):
        original_latent = original_latent["samples"] if original_latent is not None else None

        sampler = KSAMPLER(sampler_lcm_rcfg, {"enable": enable, "delta":delta, "cfg":cfg, "original_latent":original_latent})
        return (sampler, )
    
NODE_CLASS_MAPPINGS = {
    "LCMSamplerRCFG": LCMSamplerRCFG,
}