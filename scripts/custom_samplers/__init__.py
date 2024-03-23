from .gradual_latent import GradualLatentSampler
from .lcm_sampler_rcfg import LCMSamplerRCFG
from .tcd_sampler import TCDSampler
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"GradualLatentSampler{NODE_SURFIX}": GradualLatentSampler,
    f"LCMSamplerRCFG{NODE_SURFIX}": LCMSamplerRCFG,
    f"TCDSampler{NODE_SURFIX}": TCDSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"GradualLatentSampler{NODE_SURFIX}": f"Gradual Latent Sampler {SYMBOL}",
    f"LCMSamplerRCFG{NODE_SURFIX}": f"LCM Sampler RCFG {SYMBOL}",
    f"TCDSampler{NODE_SURFIX}": f"TCD Sampler {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]