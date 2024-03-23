from .node import CDTuner
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"CD_Tuner{NODE_SURFIX}": CDTuner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"CD_Tuner{NODE_SURFIX}": f"CD Tuner {SYMBOL}"
}