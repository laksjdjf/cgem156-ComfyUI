import comfy
from ... import ROOT_NAME
import math

CATEGORY_NAME = ROOT_NAME + "/custom_noise"

class VariationNoiseGenarator:
    def __init__(self, seed, variation_seed, similarity, batch_index):
        self.seed = seed
        self.variation_seed = variation_seed
        self.similarity = similarity
        self.batch_index = batch_index

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        base_noise = comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)
        variation_noise = comfy.sample.prepare_noise(latent_image, self.variation_seed, batch_inds)
        noise = base_noise[self.batch_index].unsqueeze(0) * self.similarity  + variation_noise * math.sqrt(1 - self.similarity ** 2)
        return noise

class VariationNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "similarity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "batch_index": ("INT", {"default": 1, "min": 0, "max": 4096}),
            }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = CATEGORY_NAME

    def get_noise(self, base_seed, seed, similarity, batch_index):
        return (VariationNoiseGenarator(base_seed, seed, similarity, batch_index-1),)