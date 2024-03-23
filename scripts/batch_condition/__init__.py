from .node import CLIPTextEncodeBatch, StringInput, BatchString, PrefixString, SaveBatchString, SaveImageBatch, SaveLatentBatch
from ... import NODE_SURFIX, SYMBOL

NODE_CLASS_MAPPINGS = {
    f"CLIPTextEncodeBatch{NODE_SURFIX}": CLIPTextEncodeBatch,
    f"StringInput{NODE_SURFIX}": StringInput,
    f"BatchString{NODE_SURFIX}": BatchString,
    f"PrefixString{NODE_SURFIX}": PrefixString,
    f"SaveBatchString{NODE_SURFIX}": SaveBatchString,
    f"SaveImageBatch{NODE_SURFIX}": SaveImageBatch,
    f"SaveLatentBatch{NODE_SURFIX}": SaveLatentBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"CLIPTextEncodeBatch{NODE_SURFIX}": f"CLIP Text Encode Batch {SYMBOL}",
    f"StringInput{NODE_SURFIX}": f"String Input {SYMBOL}",
    f"BatchString{NODE_SURFIX}": f"Batch String {SYMBOL}",
    f"PrefixString{NODE_SURFIX}": f"Prefix String {SYMBOL}",
    f"SaveBatchString{NODE_SURFIX}": f"Save Batch String {SYMBOL}",
    f"SaveImageBatch{NODE_SURFIX}": f"Save Image Batch {SYMBOL}",
    f"SaveLatentBatch{NODE_SURFIX}": f"Save Latent Batch {SYMBOL}"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]