from .reference import ReferenceApply, ReferenceLatent
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"ReferenceApply{NODE_SURFIX}": ReferenceApply,
    f"ReferenceLatent{NODE_SURFIX}": ReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"ReferenceApply{NODE_SURFIX}": f"Reference Apply {SYMBOL}",
    f"ReferenceLatent{NODE_SURFIX}": f"Reference Latent {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]