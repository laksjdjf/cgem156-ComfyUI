from .node import ScaleCrafter
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"ScaleCrafter{NODE_SURFIX}": ScaleCrafter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"ScaleCrafter{NODE_SURFIX}": f"Scale Crafter {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]