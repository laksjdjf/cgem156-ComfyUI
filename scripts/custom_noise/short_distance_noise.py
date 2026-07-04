import comfy
from ... import ROOT_NAME, SYMBOL, NODE_SURFIX
import torch
from comfy_api.v0_0_2 import io
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

class ShortDistanceNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"ShortDistanceNoise{NODE_SURFIX}",
            display_name=f"Short Distance Noise {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("num_samples", default=32, min=1, max=4096),
                io.Latent.Input("reference_latents"),
            ],
            outputs=[
                io.Noise.Output(),
            ],
        )

    @classmethod
    def execute(cls, seed, num_samples, reference_latents) -> io.NodeOutput:
        return io.NodeOutput(Noise_ShortDistance(seed, reference_latents))

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

class SameColorNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"SameColorNoise{NODE_SURFIX}",
            display_name=f"Same Color Noise {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Latent.Input("reference_latents"),
                io.Float.Input("strength", default=0.1, min=-1.0, max=1.0, step=0.01),
                *[io.Boolean.Input(f"ch_{i:02d}", default=True) for i in range(16)],
            ],
            outputs=[
                io.Noise.Output(),
            ],
        )

    @classmethod
    def execute(cls, seed, reference_latents, strength, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(Noise_SameColor(seed, reference_latents, strength, **kwargs))
