from .node import LoadTagger, PredictTag, GradCam, GradCamAuto, GradPair, WDTaggerSimilarity
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"LoadTagger{NODE_SURFIX}": LoadTagger,
    f"PredictTag{NODE_SURFIX}": PredictTag,
    f"GradCam{NODE_SURFIX}": GradCam,
    f"GradCamAuto{NODE_SURFIX}": GradCamAuto,
    f"GradPair{NODE_SURFIX}": GradPair,
    f"WDTaggerSimilarity{NODE_SURFIX}": WDTaggerSimilarity,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"LoadTagger{NODE_SURFIX}": f"Load Tagger {SYMBOL}",
    f"PredictTag{NODE_SURFIX}": f"Predict Tag {SYMBOL}",
    f"GradCam{NODE_SURFIX}": f"Grad Cam {SYMBOL}",
    f"GradCamAuto{NODE_SURFIX}": f"Grad Cam Auto {SYMBOL}",
    f"GradPair{NODE_SURFIX}": f"Grad Pair {SYMBOL}",
    f"WDTaggerSimilarity{NODE_SURFIX}": f"WD Tagger Similarity {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
