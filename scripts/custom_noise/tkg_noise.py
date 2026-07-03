import comfy
from typing import NamedTuple
import torch
import torch.nn.functional as F
from ... import ROOT_NAME, SYMBOL, NODE_SURFIX
from comfy_api.v0_0_2 import io
CATEGORY_NAME = ROOT_NAME + "custom_noise"

def get_mean_shifted_latents(
    latents: torch.Tensor,
    shift: float = 0.11,
    delta_shift: float = 0.1,
    channels: list[float] = [0, 1, 1, 0],  # list of {-1, 0, 1}
) -> torch.Tensor:
    shifted_latents = latents.clone()

    for idx, sign in enumerate(channels):
        if sign == 0:
            # skip
            continue

        latent_channel = shifted_latents[:, idx, :, :]

        positive_ratio = (latent_channel > 0).float().mean()
        target_ratio = positive_ratio + shift * sign

        # gradually shift latent_channel
        while True:
            latent_channel += delta_shift * sign
            new_positive_ratio = (latent_channel > 0).float().mean()
            if new_positive_ratio >= target_ratio:
                break

        # replace the channel in the original latents
        shifted_latents[:, idx, :, :] = latent_channel

    return shifted_latents


def get_2d_gaussian(
    latent_height: int,
    latent_width: int,
    std_dev: float,
    device: torch.device,
    center_x: float = 0.0,
    center_y: float = 0.0,
    factor: int = 8,  # idk why
):
    y = torch.linspace(-1, 1, steps=latent_height // factor, device=device)
    x = torch.linspace(-1, 1, steps=latent_width // factor, device=device)

    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

    x_grid = x_grid - center_x
    y_grid = y_grid - center_y

    gauss = torch.exp(-((x_grid**2 + y_grid**2) / (2 * std_dev**2)))
    gauss = gauss[None, None, :, :]  # add batch and channel dimensions

    return gauss


def apply_tkg_noise(
    latents: torch.Tensor,
    shift: float = 0.11,
    delta_shift: float = 0.1,
    std_dev: float = 0.5,
    factor: int = 8,
    channels: list[float] = [0, 1, 1, 0],
):
    batch_size, num_channels, latent_height, latent_width = latents.shape

    shifted_latents = get_mean_shifted_latents(
        latents,
        shift=shift,
        delta_shift=delta_shift,
        channels=channels,
    )
    gauss_mask = get_2d_gaussian(
        latent_height=latent_height,
        latent_width=latent_width,
        std_dev=std_dev,
        center_x=0.0,
        center_y=0.0,
        factor=factor,
        device=latents.device,
    )
    gauss_mask = F.interpolate(
        gauss_mask,
        size=(latent_height, latent_width),
        mode="bilinear",
        align_corners=False,
    )

    gauss_mask = gauss_mask.expand(batch_size, num_channels, -1, -1)

    noised_latents = shifted_latents * (1 - gauss_mask) + latents * gauss_mask

    return noised_latents


class ColorSet(NamedTuple):
    name: str
    channels: list[float]


# ref: Figure 28. Additional Result in various color Background with SD
COLOR_SETS: list[ColorSet] = [
    ColorSet("green", [0, 1, 1, 0]),
    ColorSet("cyan", [0, 1, 0, 0]),
    ColorSet("magenta", [0, -1, -1, -1]),
    ColorSet("purple", [0, 0, -1, -1]),
    ColorSet("black", [-1, 0, 0, 1]),
    ColorSet("orange", [-1, -1, 1, 0]),
    ColorSet("white", [0, 0, 0, -1]),
    ColorSet("yellow", [0, -1, 1, -1]),
]

COLOR_SET_MAP: dict[str, ColorSet] = {c.name: c for c in COLOR_SETS}

class Noise_RandomNoise:
    def __init__(self, seed, color="green", shift=0.11, grid_factor=8):
        self.seed = seed
        self.color = color
        self.shift = shift
        self.grid_factor = grid_factor

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        noise = comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)
        color_set = COLOR_SET_MAP.get(self.color, COLOR_SET_MAP["green"])
        noise = apply_tkg_noise(
            noise,
            shift=self.shift,
            channels=color_set.channels,
            factor=self.grid_factor,
        )
        return noise

class TKGRandomNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"TKGRandomNoise{NODE_SURFIX}",
            display_name=f"TKG Random Noise {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Int.Input(
                    "noise_seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    control_after_generate=True,
                ),
                io.Combo.Input(
                    "color",
                    options=[c.name for c in COLOR_SETS],
                    default="green",
                ),
                io.Float.Input(
                    "shift",
                    default=0.11,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Int.Input(
                    "grid_factor",
                    default=8,
                    min=1,
                    max=16,
                    step=1,
                ),
            ],
            outputs=[
                io.Noise.Output(),
            ],
        )

    @classmethod
    def execute(cls, noise_seed, color, shift, grid_factor) -> io.NodeOutput:
        return io.NodeOutput(Noise_RandomNoise(noise_seed, color, shift, grid_factor))
