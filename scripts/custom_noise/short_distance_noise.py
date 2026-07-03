import comfy
from ... import ROOT_NAME
import torch
CATEGORY_NAME = ROOT_NAME + "custom_noise"
    
class Noise_ShortDistance:
    def __init__(self, seed, num_samples=32, reference_latents=None):
        self.seed = seed
        self.num_samples = num_samples
        self.reference_latents = reference_latents

    def generate_noise(self, input_latent):
        assert self.reference_latents["samples"].shape == input_latent["samples"].shape, "Reference latents and input latents must have the same shape."
        latent = self.reference_latents["samples"].to(input_latent["samples"].device, dtype=input_latent["samples"].dtype)
        batch_inds = input_latent.get("batch_index", None)
        B = latent.shape[0]
        K = self.num_samples

        latent_repeat = latent.unsqueeze(1).repeat(1, K, *[1 for _ in latent.shape[1:]])
        noise = comfy.sample.prepare_noise(latent_repeat, self.seed, batch_inds)

        diff = (latent_repeat - noise) ** 2
        dist = diff.flatten(start_dim=2).sum(dim=2)
        best_idx = dist.argmin(dim=1)

        # gatherで最短ノイズを選択
        idx_expand = best_idx.view(B, 1, *[1 for _ in latent.shape[1:]]).expand_as(latent_repeat[:, :1])
        best_noise = noise.gather(1, idx_expand).squeeze(1)

        return best_noise
    
class ShortDistanceNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required":{
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "num_samples": ("INT", {"default": 32, "min": 1, "max": 4096}),
                    "reference_latents": ("LATENT", ),
            }
        }
    
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = CATEGORY_NAME

    def get_noise(self, seed, reference_latents):
        return (Noise_ShortDistance(seed, reference_latents),)
    
class Noise_SameColor:
    def __init__(self, seed, reference_latents, strength, **kwargs):
        self.seed = seed
        self.reference_latents = reference_latents
        self.strength = strength
        self.channel_mask = torch.tensor([1.0 if kwargs.get(f"ch_{i:02d}", True) else 0.0 for i in range(16)])

    def generate_noise(self, input_latent):
        assert self.reference_latents["samples"].shape == input_latent["samples"].shape, "Reference latents and input latents must have the same shape."
        latent = self.reference_latents["samples"].to(input_latent["samples"].device, dtype=input_latent["samples"].dtype)
        batch_inds = input_latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent, self.seed, batch_inds)

        latent_mean = latent.mean(dim=1, keepdim=True)
        noise_mean = noise.mean(dim=1, keepdim=True)
        channel_mask = self.channel_mask.to(latent.device, dtype=latent.dtype).view(1, -1, *[1 for _ in range(len(latent.shape)-2)])
        
        noise = noise + (latent_mean - noise_mean) * self.strength * channel_mask

        return noise
    
class SameColorNoise:
    @classmethod
    def INPUT_TYPES(s):
        retval = {
                "required":{
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "reference_latents": ("LATENT", ),
                    "strength": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

        for i in range(16):
            retval["required"][f"ch_{i:02d}"] = ("BOOLEAN", {"default": True})
        return retval
    
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = CATEGORY_NAME

    def get_noise(self, seed, reference_latents, strength, **kwargs):
        return (Noise_SameColor(seed, reference_latents, strength, **kwargs),)