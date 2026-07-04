import comfy
from latent_preview import get_previewer
import numpy as np
import torch
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, SYMBOL, NODE_SURFIX

def image_to_tensor(image):
    return torch.tensor(np.array(image).astype(np.float32)) / 255.0

def prepare_callback(model, steps, x0_output_dict=None, previews=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        nonlocal previews

        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            previews.append(image_to_tensor(preview_bytes[1]))
        
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

class SamplerCustomAdvancedPreview(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"SamplerCustomAdvancedPreview{NODE_SURFIX}",
            display_name=f"Sampler Custom Advanced Preview {SYMBOL}",
            category=ROOT_NAME + "custom_samplers",
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
                io.Image.Output(display_name="previews"),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image) -> io.NodeOutput:
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        previews = []
        callback = prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output, previews)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out

        previews = torch.stack(previews)
        return io.NodeOutput(out, out_denoised, previews)