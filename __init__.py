import importlib
import os

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
    "for_test",
    "lora_xy",
    "reference",
]

def import_from_package(module_name, module_path):
    if os.path.isdir(module_path):
        module_file = os.path.join(module_path, "__init__.py")
    else:
        module_file = module_path

    if not os.path.exists(module_file):
        raise FileNotFoundError(f"{module_file} not found")

    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    import timm
except ImportError:
    print("timm is not installed. Skipping load wd-tagger.")
else:
    scripts.append("wd-tagger")

for script in scripts:
    #module = importlib.import_module(f"custom_nodes.cgem156-ComfyUI.scripts.{script}")
    module = import_from_package(f"custom_nodes.cgem156-ComfyUI.scripts.{script}", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", script))
    if hasattr(module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(getattr(module, 'NODE_CLASS_MAPPINGS'))
    if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'))

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
