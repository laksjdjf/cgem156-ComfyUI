from .gradual_latent import GradualLatentSampler
from .lcm_sampler_rcfg import LCMSamplerRCFG
from .tcd_sampler import TCDSampler
from .sampler_custom_preview import SamplerCustomAdvancedPreview
from .euler_ancestral_fixed_noise import SamplerEulerAncestralFixedNoise

from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"GradualLatentSampler{NODE_SURFIX}": GradualLatentSampler,
    f"LCMSamplerRCFG{NODE_SURFIX}": LCMSamplerRCFG,
    f"TCDSampler{NODE_SURFIX}": TCDSampler,
    f"SamplerCustomAdvancedPreview{NODE_SURFIX}": SamplerCustomAdvancedPreview,
    f"SamplerEulerAncestralFixedNoise{NODE_SURFIX}": SamplerEulerAncestralFixedNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"GradualLatentSampler{NODE_SURFIX}": f"Gradual Latent Sampler {SYMBOL}",
    f"LCMSamplerRCFG{NODE_SURFIX}": f"LCM Sampler RCFG {SYMBOL}",
    f"TCDSampler{NODE_SURFIX}": f"TCD Sampler {SYMBOL}",
    f"SamplerCustomAdvancedPreview{NODE_SURFIX}": f"Sampler Custom Advanced Preview {SYMBOL}",
    f"SamplerEulerAncestralFixedNoise{NODE_SURFIX}": f"Sampler Euler Ancestral Fixed Noise {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]