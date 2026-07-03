from .variation_noise import VariationNoise, RandomNoiseOffset, RandomNoiseVariationSimple
from .short_distance_noise import ShortDistanceNoise, SameColorNoise
from .tkg_noise import TKGRandomNoise
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"VariationNoise{NODE_SURFIX}": VariationNoise,
    f"RandomNoiseOffset{NODE_SURFIX}": RandomNoiseOffset,
    f"RandomNoiseVariationSimple{NODE_SURFIX}": RandomNoiseVariationSimple,
    f"TKGRandomNoise{NODE_SURFIX}": TKGRandomNoise,
    f"ShortDistanceNoise{NODE_SURFIX}": ShortDistanceNoise,
    f"SameColorNoise{NODE_SURFIX}": SameColorNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"VariationNoise{NODE_SURFIX}": f"Variation Noise {SYMBOL}",
    f"RandomNoiseOffset{NODE_SURFIX}": f"Random Noise Offset {SYMBOL}",
    f"RandomNoiseVariationSimple{NODE_SURFIX}": f"Random Noise Variation Simple {SYMBOL}",
    f"TKGRandomNoise{NODE_SURFIX}": f"TKG Random Noise {SYMBOL}",
    f"ShortDistanceNoise{NODE_SURFIX}": f"Short Distance Noise {SYMBOL}",
    f"SameColorNoise{NODE_SURFIX}": f"Same Color Noise {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]