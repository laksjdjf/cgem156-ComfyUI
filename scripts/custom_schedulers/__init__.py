from .text_scheduler import TextScheduler
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"TextScheduler{NODE_SURFIX}": TextScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"TextScheduler{NODE_SURFIX}": f"Text Scheduler {SYMBOL}",
}