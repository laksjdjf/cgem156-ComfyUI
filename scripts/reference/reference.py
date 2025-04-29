import torch
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "reference"

def new_vec(mode, chunks, x, index):
    xs = x.clone().chunk(chunks, dim=0)
    ref_xs = torch.cat([xi[index].unsqueeze(0).expand(xi.shape[0], -1, -1).clone() for xi in xs], dim=0).clone()

    if mode == "concat":
        new_x = x.clone()
        return torch.cat([new_x, ref_xs], dim=1)
    else:
        return ref_xs

class ReferenceApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL",),
                "index": ("INT", {"default": 0, "min": 0, "max": 256}),
                "mode": (["concat", "replace"], {"default": "concat"}),
                "depth": ("INT", {"default": 12, "min": -1, "max": 12}),
                "start_step": ("FLOAT", {"default": 0,"min": 0, "max": 1, "step": 0.01}),
                "end_step": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "apply_input": ("BOOLEAN", {"default": True}),
                "apply_middle": ("BOOLEAN", {"default": True}),
                "apply_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "reference_only"

    CATEGORY = CATEGORY_NAME

    def reference_only(self, model, index, mode, depth, start_step, end_step, apply_input, apply_middle, apply_output): 
        model_reference = model.clone()
        start_sigma = model_reference.model.model_sampling.percent_to_sigma(start_step)
        end_sigma = model_reference.model.model_sampling.percent_to_sigma(end_step)

        self.depth = depth

        self.sdxl = hasattr(model_reference.model.diffusion_model, "label_emb")
        self.num_blocks = 8 if self.sdxl else 11

        def reference_apply(q, k, v, extra_options):
            block_name, block_id = extra_options["block"]
            chunks = len(extra_options["cond_or_uncond"])

            if block_name == "input" and not apply_input:
                return q, k, v
            if block_name == "middle" and not apply_middle:
                return q, k, v
            if block_name == "output" and not apply_output:
                return q, k, v
            
            if block_name == "output":
                block_number = self.num_blocks - block_id
            else:
                block_number = block_id

            q_out = q.clone()
            k_out = k.clone()
            v_out = v.clone()

            sigma = extra_options["sigmas"][0].item()

            if end_sigma <= sigma <= start_sigma and block_number <= self.depth:
                k_out = new_vec(mode, chunks, k_out, index)
                v_out = new_vec(mode, chunks, v_out, index)

            return q_out, k_out, v_out

        model_reference.set_model_attn1_patch(reference_apply)

        return (model_reference, )
    
class ReferenceLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "latent": ("LATENT",),
                "index": ("INT", {"default": 0, "min": 0, "max": 256}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 256}),
            }
        }
    
    RETURN_TYPES = ("LATENT", )
    FUNCTION = "reference_latent"
    CATEGORY = CATEGORY_NAME

    def reference_latent(self, latent, index, batch_size):
        latent_new = latent.copy()

        sample = latent_new["samples"]
        height, width = sample.shape[2], sample.shape[3]

        empty_latent = torch.zeros_like(latent["samples"]).repeat(batch_size , 1, 1, 1)
        empty_latent[index] = sample[0]
        noise_mask = torch.ones(batch_size, 1, height * 8, width * 8).to(sample)
        noise_mask[index] = 0.0

        latent_new["samples"] = empty_latent
        latent_new["noise_mask"] = noise_mask

        return (latent_new, )

