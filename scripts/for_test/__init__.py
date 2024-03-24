from .attention_scale import AttentionScale
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"AttentionScale{NODE_SURFIX}": AttentionScale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"AttentionScale{NODE_SURFIX}": f"Attention Scale {SYMBOL}"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]