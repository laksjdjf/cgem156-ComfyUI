from .variation_noise import VariationNoise
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"VariationNoise{NODE_SURFIX}": VariationNoise
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"VariationNoise{NODE_SURFIX}": f"Variation Noise {SYMBOL}"
}
