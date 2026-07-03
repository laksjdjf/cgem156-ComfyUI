from .attention_scale import AttentionScale
from .kv_token_multiplier import CLIPTextEncodeBatchKVMultiply
from .kmeans_quant import KmeansQuantize
from .mse_heatmap import MSEHeatmap, MSEHeatmapTagger
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"AttentionScale{NODE_SURFIX}": AttentionScale,
    f"CLIPTextEncodeBatchKVMultiply{NODE_SURFIX}": CLIPTextEncodeBatchKVMultiply,
    f"KmeansQuantize{NODE_SURFIX}": KmeansQuantize,
    f"MSEHeatmap{NODE_SURFIX}": MSEHeatmap,
    f"MSEHeatmapTagger{NODE_SURFIX}": MSEHeatmapTagger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"AttentionScale{NODE_SURFIX}": f"Attention Scale {SYMBOL}",
    f"CLIPTextEncodeBatchKVMultiply{NODE_SURFIX}": f"CLIP Text Encode Batch KV Multiply {SYMBOL}",
    f"KmeansQuantize{NODE_SURFIX}": f"Kmeans Quantize {SYMBOL}",
    f"MSEHeatmap{NODE_SURFIX}": f"MSE Heatmap {SYMBOL}",
    f"MSEHeatmapTagger{NODE_SURFIX}": f"MSE Heatmap Tagger {SYMBOL}"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]