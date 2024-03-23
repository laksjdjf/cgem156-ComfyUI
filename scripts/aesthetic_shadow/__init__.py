from .node import LoadAestheticShadow, PredictAesthetic
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"LoadAestheticShadow{NODE_SURFIX}": LoadAestheticShadow,
    f"PredictAesthetic{NODE_SURFIX}": PredictAesthetic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"LoadAestheticShadow{NODE_SURFIX}": f"Load Aesthetic Shadow {SYMBOL}",
    f"PredictAesthetic{NODE_SURFIX}": f"Predict Aesthetic {SYMBOL}",
}
