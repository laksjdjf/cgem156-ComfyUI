from .node import LortnocLoader
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"LortnocLoader{NODE_SURFIX}": LortnocLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"LortnocLoader{NODE_SURFIX}": f"Lortnoc Loader {SYMBOL}"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]