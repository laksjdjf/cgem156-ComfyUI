import torch
from comfy.ldm.modules.attention import optimized_attention
from ... import ROOT_NAME

def attention_pytorch(q, k, v, heads, temperature=1.0, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(
        lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
        (q, k, v),
    )

    scale = (dim_head ** -0.5) / temperature

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale)
    out = (
        out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    )
    return out

class AttentionScale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "temperature": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "start_step": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.001}),
                "end_step": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.001}),
                "attn1": ("BOOLEAN", {"default": True}),
                "attn2": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "apply"
    CATEGORY = ROOT_NAME + "for_test"

    def apply(self, model, temperature, start_step, end_step, attn1, attn2):
        new_model = model.clone()

        self.temperature = temperature
        self.start_sigma = new_model.model.model_sampling.percent_to_sigma(start_step)
        self.end_sigma = new_model.model.model_sampling.percent_to_sigma(end_step)

        def attn_patch(q, k, v, extra_options):
            sigma = extra_options["sigmas"][0].item()

            if self.end_sigma <= sigma <= self.start_sigma:
                output = attention_pytorch(q, k, v, extra_options["n_heads"], temperature = self.temperature)
            else:
                output = attention_pytorch(q, k, v, extra_options["n_heads"], temperature = 1.0)

            return output
        
        def dummy_attn_path(q, k, v, extra_options):
            return optimized_attention(q, k, v, extra_options["n_heads"])

        self.sdxl = hasattr(new_model.model.diffusion_model, "label_emb")

        attn1_patch = attn_patch if attn1 else dummy_attn_path
        attn2_patch = attn_patch if attn2 else dummy_attn_path

        if not self.sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                new_model.set_model_attn1_replace(attn1_patch, "input", id)
                new_model.set_model_attn2_replace(attn2_patch, "input", id)
            new_model.set_model_attn1_replace(attn1_patch, "middle", 0)
            new_model.set_model_attn2_replace(attn2_patch, "middle", 0)
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                new_model.set_model_attn1_replace(attn1_patch, "output", id)
                new_model.set_model_attn2_replace(attn2_patch, "output", id)
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    new_model.set_model_attn1_replace(attn1_patch, "input", id, index)
                    new_model.set_model_attn2_replace(attn2_patch, "input", id, index)
            for index in range(10):
                new_model.set_model_attn1_replace(attn1_patch, "middle", 0, index)
                new_model.set_model_attn2_replace(attn2_patch, "middle", 0, index)
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    new_model.set_model_attn1_replace(attn1_patch, "output", id, index)
                    new_model.set_model_attn2_replace(attn2_patch, "output", id, index)

        return (new_model, )

