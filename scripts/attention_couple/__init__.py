from .node import AttentionCouple
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"AttentionCouple{NODE_SURFIX}": AttentionCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"AttentionCouple{NODE_SURFIX}": f"Attention Couple {SYMBOL}"
}

