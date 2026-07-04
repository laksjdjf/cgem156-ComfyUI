from .node import create_class, MultipleLoraLoaderDynamic
import os
from ... import SYMBOL, NODE_SURFIX

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(CURRENT_DIR, "config.txt"), "r") as f:
    config = f.read()
num_loras = [int(i) for i in config.replace(" ", "").split(",")]

NODE_CLASS_MAPPINGS = {
    f"MultipleLoraLoader{i}{NODE_SURFIX}": create_class(i) for i in num_loras
}
NODE_CLASS_MAPPINGS[f"MultipleLoraLoaderDynamic{NODE_SURFIX}"] = MultipleLoraLoaderDynamic

NODE_DISPLAY_NAME_MAPPINGS = {
    f"MultipleLoraLoader{i}{NODE_SURFIX}": f"MultipleLoraLoader{i} {SYMBOL}" for i in num_loras
}
NODE_DISPLAY_NAME_MAPPINGS[f"MultipleLoraLoaderDynamic{NODE_SURFIX}"] = f"MultipleLoraLoaderDynamic {SYMBOL}"



__all__ = [NODE_CLASS_MAPPINGS]