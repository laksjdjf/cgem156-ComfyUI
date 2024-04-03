from .node import LoraLoaderModelOnlyXY, SamplerCustomXY, KSamplerXY, KSamplerAdvancedXY, PreviewXY
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"LoraLoaderModelOnlyXY{NODE_SURFIX}": LoraLoaderModelOnlyXY,
    f"SamplerCustomXY{NODE_SURFIX}": SamplerCustomXY,
    f"KSamplerXY{NODE_SURFIX}": KSamplerXY,
    f"KSamplerAdvancedXY{NODE_SURFIX}": KSamplerAdvancedXY,
    f"PreviewXY{NODE_SURFIX}": PreviewXY
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"LoraLoaderModelOnlyXY{NODE_SURFIX}": f"Lora Loader Model Only XY {SYMBOL}",
    f"SamplerCustomXY{NODE_SURFIX}": f"Sampler Custom XY {SYMBOL}",
    f"KSamplerXY{NODE_SURFIX}": f"KSampler XY {SYMBOL}",
    f"KSamplerAdvancedXY{NODE_SURFIX}": f"KSampler Advanced XY {SYMBOL}",
    f"PreviewXY{NODE_SURFIX}": f"Preview XY {SYMBOL}"
}

