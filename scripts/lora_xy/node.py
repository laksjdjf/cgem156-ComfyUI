import comfy
from comfy_extras.nodes_custom_sampler import SamplerCustom
import folder_paths
from nodes import LoraLoader, PreviewImage, KSampler, KSamplerAdvanced
from ... import ROOT_NAME
import torch
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os

CATEGORY_NAME = ROOT_NAME + "lora_xy"

class LoraLoaderModelOnlyXY(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_list": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("XY_MODEL","XY_LIST", )
    FUNCTION = "load_lora_model_only_xy"
    CATEGORY = CATEGORY_NAME


    def load_lora_model_only_xy(self, model, lora_name, strength_list):
        models = []
        xy_list = []

        weights = [float(x.strip()) for x in strength_list.strip().strip(",").split(",")]
        for value in weights:
            models.append(self.load_lora(model, None, lora_name, value, 0)[0])
            xy_list.append(f"{lora_name.split('.')[0]}:{value}")

        return (models, xy_list)

class SamplerCustomXY(SamplerCustom):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model_xy": ("XY_MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    FUNCTION = "sample_xy"
    CATEGORY = CATEGORY_NAME

    def sample_xy(self, model_xy, **kwargs):
        outputs = []
        denoised_outputs = []

        for model in model_xy:
            output, denoised_output = self.sample(model, **kwargs)
            outputs.append(output["samples"])
            denoised_outputs.append(denoised_output["samples"])

        return ({"samples":torch.cat(outputs)}, {"samples":torch.cat(denoised_outputs)})
    
class KSamplerXY(KSampler):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model_xy": ("XY_MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }
    
    FUNCTION = "sample_xy"
    CATEGORY = CATEGORY_NAME

    def sample_xy(self, model_xy, **kwargs):
        outputs = []

        for model in model_xy:
            output = self.sample(model, **kwargs)[0]
            outputs.append(output["samples"])

        return ({"samples":torch.cat(outputs)},)
    
class KSamplerAdvancedXY(KSamplerAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model_xy": ("XY_MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }
    
    FUNCTION = "sample_xy"
    CATEGORY = CATEGORY_NAME

    def sample_xy(self, model_xy, **kwargs):
        outputs = []

        for model in model_xy:
            output = self.sample(model, **kwargs)[0]
            outputs.append(output["samples"])

        return ({"samples":torch.cat(outputs)},)
    

class PreviewXY(PreviewImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{"images": ("IMAGE", ), "xy_list": ("XY_LIST", )},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    CATEGORY_NAME = ROOT_NAME + "preview_xy"
    
    def save_images(self, images, xy_list, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        
        pil_images = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        img = self.xy_plot(pil_images, xy_list)
        

        file = "lora_xy_.png"
        img.save(os.path.join(full_output_folder, file), compress_level=self.compress_level)
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

        return { "ui": { "images": results } }
    
    def xy_plot(self, images, xy_list, text_height=100):
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

        text_images = [self.text_to_image(title).resize((image_width, text_height)) for title in xy_list]

        # 文字列の画像をキャンバスに配置（各列の上部に）
        for i, text_img in enumerate(text_images):
            canvas.paste(text_img, (i * image_width, 0))

        return canvas

    def text_to_image(self, text):
        font = ImageFont.load_default()
        img = Image.new('RGB', (256, 20), 'white')
        draw = ImageDraw.Draw(img)
        text_width, text_height = draw.textbbox((0,0), text, font=font)[2:]
        text_x = (256 - text_width) / 2
        text_y = (20 - text_height) / 2
        draw.text((text_x, text_y), text, font=font, fill='black')
        return img
