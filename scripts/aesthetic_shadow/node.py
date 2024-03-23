# https://huggingface.co/shadowlilac/aesthetic-shadow-v2

from transformers import pipeline
import torch
from PIL import Image
from comfy.ldm.modules.attention import optimized_attention
    
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "aeshtetic-shadow"

def optimized_forward(self):
    def forward(hidden_states, head_mask = None, output_attentions = False):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        context_layer = optimized_attention(query, key, value, self.num_attention_heads, head_mask)
        outputs = (context_layer, None) if output_attentions else (context_layer,)

        return outputs
    
    return forward

def optimize(model):
    for module in model.modules():
        if module.__class__.__name__ == "ViTSelfAttention":
            module.forward = optimized_forward(module)

class LoadAestheticShadow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "shadowlilac/aesthetic-shadow-v2"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "optimize_attention": ("BOOLEAN", {"default": False})
            }
        }
    RETURN_TYPES = ("AESTHETIC_SHADOW_MODEL", )
    FUNCTION = "load"

    CATEGORY = CATEGORY_NAME

    def load(self, model, device, optimize_attention):
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = pipeline("image-classification", model=model, device=device, torch_dtype=dtype)
        if optimize_attention:
            optimize(pipe.model)
        return (pipe, )

class PredictAesthetic:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "model": ("AESTHETIC_SHADOW_MODEL", ),
            },
        }
    RETURN_TYPES = ("STRING", )
    FUNCTION = "predict"

    CATEGORY = CATEGORY_NAME

    def predict(self, image, model):
        images = (image * 255).numpy().astype('uint8')
        images = [Image.fromarray(image) for image in images]
        results = []
        for image in images: # avoide batch processing
            result = model(images=[image])
            if result[0][0]["label"] == "hq":
                results.append(result[0][0]["score"])
            else:
                results.append(result[0][1]["score"])
        string = "\n".join([f"image_{i+1}:{result:4f}" for i, result in enumerate(results)])
        return (string, )

NODE_CLASS_MAPPINGS = {
    "LoadAestheticShadow": LoadAestheticShadow,
    "PredictAesthetic": PredictAesthetic
}

__all__ = ["NODE_CLASS_MAPPINGS"]