from .limited_interval_cfg_guider import LimitedIntervalCFGGuider
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"LimitedIntervalCFGGuider{NODE_SURFIX}": LimitedIntervalCFGGuider
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"LimitedIntervalCFGGuider{NODE_SURFIX}": f"Limited Interval CFG Guider {SYMBOL}"
}

