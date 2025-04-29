from .variation_noise import VariationNoise, RandomNoiseOffset, RandomNoiseVariationSimple
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"VariationNoise{NODE_SURFIX}": VariationNoise,
    f"RandomNoiseOffset{NODE_SURFIX}": RandomNoiseOffset,
    f"RandomNoiseVariationSimple{NODE_SURFIX}": RandomNoiseVariationSimple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"VariationNoise{NODE_SURFIX}": f"Variation Noise {SYMBOL}",
    f"RandomNoiseOffset{NODE_SURFIX}": f"Random Noise Offset {SYMBOL}",
    f"RandomNoiseVariationSimple{NODE_SURFIX}": f"Random Noise Variation Simple {SYMBOL}"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]