from comfy.samplers import KSAMPLER
from comfy.k_diffusion.sampling import sample_euler_ancestral
import torch
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, SYMBOL, NODE_SURFIX

def fixed_noise_sampler(x, seed=None):
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)[:1]

@torch.no_grad()
def sample_euler_ancestral_fixed_noise(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    noise_sampler = fixed_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    return sample_euler_ancestral(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler)

class SamplerEulerAncestralFixedNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"SamplerEulerAncestralFixedNoise{NODE_SURFIX}",
            display_name=f"Sampler Euler Ancestral Fixed Noise {SYMBOL}",
            category=ROOT_NAME + "custom_samplers",
            inputs=[
                io.Combo.Input("noise", options=["fixed", "random"], default="fixed"),
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False),
            ],
            outputs=[
                io.Sampler.Output(),
            ],
        )

    @classmethod
    def execute(cls, noise, eta, s_noise) -> io.NodeOutput:
        sampler = KSAMPLER(sample_euler_ancestral_fixed_noise if noise=="fixed" else sample_euler_ancestral, {"eta": eta, "s_noise": s_noise})
        return io.NodeOutput(sampler)