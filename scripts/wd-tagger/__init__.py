from .node import LoadTagger, PredictTag, GradCam, GradCamAuto
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"LoadTagger{NODE_SURFIX}": LoadTagger,
    f"PredictTag{NODE_SURFIX}": PredictTag,
    f"GradCam{NODE_SURFIX}": GradCam,
    f"GradCamAuto{NODE_SURFIX}": GradCamAuto,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"LoadTagger{NODE_SURFIX}": f"Load Tagger {SYMBOL}",
    f"PredictTag{NODE_SURFIX}": f"Predict Tag {SYMBOL}",
    f"GradCam{NODE_SURFIX}": f"Grad Cam {SYMBOL}",
    f"GradCamAuto{NODE_SURFIX}": f"Grad Cam Auto {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
