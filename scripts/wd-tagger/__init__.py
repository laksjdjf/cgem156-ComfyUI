from .node import LoadTagger, PredictTag
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"LoadTagger{NODE_SURFIX}": LoadTagger,
    f"PredictTag{NODE_SURFIX}": PredictTag,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"LoadTagger{NODE_SURFIX}": f"Load Tagger {SYMBOL}",
    f"PredictTag{NODE_SURFIX}": f"Predict Tag {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
