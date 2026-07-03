from .reference import ReferenceApply, ReferenceLatent, MultipleReferenceApply, MultipleReferenceLatent
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"ReferenceApply{NODE_SURFIX}": ReferenceApply,
    f"ReferenceLatent{NODE_SURFIX}": ReferenceLatent,
    f"MultipleReferenceApply{NODE_SURFIX}": MultipleReferenceApply,
    f"MultipleReferenceLatent{NODE_SURFIX}": MultipleReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"ReferenceApply{NODE_SURFIX}": f"Reference Apply {SYMBOL}",
    f"ReferenceLatent{NODE_SURFIX}": f"Reference Latent {SYMBOL}",
    f"MultipleReferenceApply{NODE_SURFIX}": f"Multiple Reference Apply {SYMBOL}",
    f"MultipleReferenceLatent{NODE_SURFIX}": f"Multiple Reference Latent {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]