import comfy
from ... import ROOT_NAME
import math
import torch
import numpy as np

CATEGORY_NAME = ROOT_NAME + "custom_noise"

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
                "batch_index": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = CATEGORY_NAME

    def get_noise(self, base_seed, seed, similarity, batch_index):
        return (VariationNoiseGenarator(base_seed, seed, similarity, batch_index-1),)
    
def prepare_noise(latent_image, seed, noise_inds=None, offset=0):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        noise_offset = torch.randn(list(latent_image.size())[:2] + [1,1], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        return noise + noise_offset * offset

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)

    noise_offsets = []
    for i in range(unique_inds[-1]+1):
        noise_offset = torch.randn([1] + list(latent_image.size())[1:2] + [1,1], dtype=latent_image.dtype, generator=generator, device="cpu")
        if i in unique_inds:
            noise_offsets.append(noise_offset)
    
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)

    noise_offsets = [noise_offsets[i] for i in inverse]
    noise_offsets = torch.cat(noise_offsets, axis=0)
    return noises + noise_offsets * offset

class Noise_RandomNoiseOffset:
    def __init__(self, seed, offset):
        self.seed = seed
        self.offset = offset

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return prepare_noise(latent_image, self.seed, batch_inds, self.offset)

class RandomNoiseOffset:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                     }
                }
    
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = CATEGORY_NAME

    def get_noise(self, noise_seed, offset):
        return (Noise_RandomNoiseOffset(noise_seed, offset),)
    
class Noise_RandomNoiseVariationSimple:
    def __init__(self, seed, similarity):
        self.seed = seed
        self.similarity = similarity

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        noise = comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

        noise = torch.cat([noise[:1], noise[:1] * self.similarity + noise[1:] * math.sqrt(1 - self.similarity ** 2)])
        return noise
    
class RandomNoiseVariationSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required":{
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "similarity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = CATEGORY_NAME

    def get_noise(self, seed, similarity):
        return (Noise_RandomNoiseVariationSimple(seed, similarity),)