import comfy
import comfy.samplers
import comfy.sd
import comfy.utils
from comfy_extras.nodes_custom_sampler import SamplerCustom
import nodes
import folder_paths
from ... import ROOT_NAME, NODE_SURFIX, SYMBOL
from comfy_api.v0_0_2 import io, ui
import torch
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def generate_image_matrix(images, xy_list):
    num_images = len(images)
    cols = len(xy_list)  # 列数
    rows = num_images // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()  # 1次元配列化

    for i in range(len(axes)):
        if i < num_images:
            axes[i].imshow(images[i])
            axes[i].set_title(xy_list[i], fontsize=8)
            axes[i].axis("off")
        else:
            axes[i].axis("off")  # 余ったスペースを空白にする

    plt.tight_layout()

    # Figure をバイナリデータとして保存し、PIL画像に変換
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)

CATEGORY_NAME = ROOT_NAME + "lora_xy"

# module-level cache replacing the old per-instance `self.loaded_lora` state from
# nodes.py's LoraLoader (execute() is a classmethod, no `self` to cache on).
_lora_xy_cache = {"loaded_lora": None}

def _load_lora_model_only(model, lora_name, strength_model):
    # Mirrors nodes.py LoraLoader.load_lora(model, clip=None, lora_name, strength_model, strength_clip=0).
    if strength_model == 0:
        return model

    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora = None
    lora_metadata = None
    loaded_lora = _lora_xy_cache["loaded_lora"]
    if loaded_lora is not None:
        if loaded_lora[0] == lora_path:
            lora = loaded_lora[1]
            lora_metadata = loaded_lora[2] if len(loaded_lora) > 2 else None
        else:
            _lora_xy_cache["loaded_lora"] = None

    if lora is None:
        lora, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        _lora_xy_cache["loaded_lora"] = (lora_path, lora, lora_metadata)

    model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0, lora_metadata=lora_metadata)
    return model_lora

class LoraLoaderModelOnlyXY(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"LoraLoaderModelOnlyXY{NODE_SURFIX}",
            display_name=f"Lora Loader Model Only XY {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.String.Input("strength_list", multiline=True),
            ],
            outputs=[
                io.Custom("XY_MODEL").Output(),
                io.Custom("XY_LIST").Output(),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_list) -> io.NodeOutput:
        models = []
        xy_list = []

        weights = [float(x.strip()) for x in strength_list.strip().strip(",").split(",")]
        for value in weights:
            models.append(_load_lora_model_only(model, lora_name, value))
            xy_list.append(f"{lora_name.split('.')[0]}:{value}")

        return io.NodeOutput(models, xy_list)

class SamplerCustomXY(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"SamplerCustomXY{NODE_SURFIX}",
            display_name=f"Sampler Custom XY {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("XY_MODEL").Input("model_xy"),
                io.Boolean.Input("add_noise", default=True),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
            ],
        )

    @classmethod
    def execute(cls, model_xy, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image) -> io.NodeOutput:
        outputs = []
        denoised_outputs = []

        # Composition, not inheritance: SamplerCustom is itself a V3 io.ComfyNode now, so we
        # call its public `execute` classmethod per model instead of subclassing it. This keeps
        # us in sync with upstream's noise/x0-output/nested-tensor handling without duplicating it.
        for model in model_xy:
            result = SamplerCustom.execute(
                model=model,
                add_noise=add_noise,
                noise_seed=noise_seed,
                cfg=cfg,
                positive=positive,
                negative=negative,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent_image,
            )
            output, denoised_output = result.result
            outputs.append(output["samples"])
            denoised_outputs.append(denoised_output["samples"])

        return io.NodeOutput({"samples": torch.cat(outputs)}, {"samples": torch.cat(denoised_outputs)})

class KSamplerXY(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"KSamplerXY{NODE_SURFIX}",
            display_name=f"KSampler XY {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("XY_MODEL").Input("model_xy"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_xy, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise) -> io.NodeOutput:
        outputs = []

        # Composition: nodes.common_ksampler is the stable module-level function that both
        # KSampler and KSamplerAdvanced wrap; calling it directly avoids depending on the
        # KSampler node class itself.
        for model in model_xy:
            output = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
            outputs.append(output["samples"])

        return io.NodeOutput({"samples": torch.cat(outputs)})

class KSamplerAdvancedXY(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"KSamplerAdvancedXY{NODE_SURFIX}",
            display_name=f"KSampler Advanced XY {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("XY_MODEL").Input("model_xy"),
                io.Combo.Input("add_noise", options=["enable", "disable"]),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Int.Input("start_at_step", default=0, min=0, max=10000),
                io.Int.Input("end_at_step", default=10000, min=0, max=10000),
                io.Combo.Input("return_with_leftover_noise", options=["disable", "enable"]),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_xy, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                start_at_step, end_at_step, return_with_leftover_noise) -> io.NodeOutput:
        outputs = []

        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        # Composition: same nodes.common_ksampler function that KSamplerAdvanced.sample wraps.
        for model in model_xy:
            output = nodes.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                            denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                                            force_full_denoise=force_full_denoise)[0]
            outputs.append(output["samples"])

        return io.NodeOutput({"samples": torch.cat(outputs)})

class XYImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{"images": ("IMAGE", ), "xy_list": ("XY_LIST", )},
        }

    FUNCTION = "xy_images"
    CATEGORY_NAME = ROOT_NAME

    def xy_images(self, images, xy_list):
        pil_images = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        imgs = generate_image_matrix(pil_images, xy_list)
        img = np.array(imgs).astype(np.float32) / 255.
        img = img * 2. - 1.
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        return {"images": img}

def _xy_text_to_image(text):
    font = ImageFont.load_default()
    img = Image.new('RGB', (256, 20), 'white')
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textbbox((0,0), text, font=font)[2:]
    text_x = (256 - text_width) / 2
    text_y = (20 - text_height) / 2
    draw.text((text_x, text_y), text, font=font, fill='black')
    return img

def _xy_plot(images, xy_list, text_height=100):
    n = len(xy_list)
    m = len(images) // n

    image_width, image_height = images[0].width, images[0].height

    # キャンバスのサイズを再計算（全画像が同じサイズの場合）
    canvas_width = image_width * n
    canvas_height = (image_height * m) + text_height  # 文字列の高さ分を追加

    # キャンバスを再作成
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

    # 画像と文字列の画像をキャンバスに配置（全画像が同じサイズの場合の最適化）
    for i, img in enumerate(images):
        # 画像を配置する位置を計算
        x_offset = (i // m) * image_width
        y_offset = (i % m) * (image_height) + text_height  # 文字列の高さ分をオフセットして再計算
        canvas.paste(img, (x_offset, y_offset))

    text_images = [_xy_text_to_image(title).resize((image_width, text_height)) for title in xy_list]

    # 文字列の画像をキャンバスに配置（各列の上部に）
    for i, text_img in enumerate(text_images):
        canvas.paste(text_img, (i * image_width, 0))

    return canvas

class PreviewXY(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"PreviewXY{NODE_SURFIX}",
            display_name=f"Preview XY {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Image.Input("images"),
                io.Custom("XY_LIST").Input("xy_list"),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, xy_list) -> io.NodeOutput:
        pil_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        canvas = _xy_plot(pil_images, xy_list)

        canvas_np = np.array(canvas).astype(np.float32) / 255.
        canvas_tensor = torch.from_numpy(canvas_np).unsqueeze(0)

        return io.NodeOutput(ui=ui.PreviewImage(canvas_tensor, cls=cls))
