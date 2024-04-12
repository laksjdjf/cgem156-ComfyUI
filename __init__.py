import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
SYMBOL = "🍌"
NODE_SURFIX = f"|cgem156"
ROOT_NAME = f"cgem156 {SYMBOL}/"

scripts = [
    "batch_condition", 
    "dart", 
    "lortnoc",
    "attention_couple", 
    "cd_tuner", 
    "lora_merger",
    "multiple_lora_loader",
    "custom_samplers", 
    "custom_schedulers",
    "custom_guiders",
    "custom_noise",
    "scale_crafter", 
    "aesthetic_shadow",
    "for_test",
    "lora_xy"
]

try:
    import timm
except ImportError:
    print("timm is not installed. Skipping load wd-tagger.")
else:
    scripts.append("wd-tagger")

for script in scripts:
    module = importlib.import_module(f"custom_nodes.cgem156-ComfyUI.scripts.{script}")
    if hasattr(module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(getattr(module, 'NODE_CLASS_MAPPINGS'))
    if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'))

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
