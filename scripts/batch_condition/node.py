import torch
import numpy as np
import math
import os
from PIL import Image
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, NODE_SURFIX, SYMBOL


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CATEGORY_NAME = ROOT_NAME + "batch_condition"

def lcm(a, b):
    return a * b // math.gcd(a, b)

def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm

class CLIPTextEncodeBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"CLIPTextEncodeBatch{NODE_SURFIX}",
            display_name=f"CLIP Text Encode Batch {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Clip.Input("clip"),
                io.Custom("BATCH_STRING").Input("texts"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, texts) -> io.NodeOutput:
        conds = []
        pooleds = []
        num_tokens = []
        for text in texts:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conds.append(cond)
            pooleds.append(pooled)
            num_tokens.append(cond.shape[1])

        # Make number of tokens equal
        # attn(q, k, v) == attn(q, [k]*n, [v]*n)
        lcm = lcm_for_list(num_tokens)
        repeats = [lcm//num for num in num_tokens]
        conds = torch.cat([cond.repeat(1, repeat, 1) for cond, repeat in zip(conds, repeats)])
        pooleds = torch.cat(pooleds)
        return io.NodeOutput([[conds, {"pooled_output": pooleds}]])

class StringInput(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"StringInput{NODE_SURFIX}",
            display_name=f"String Input {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.String.Input("text", multiline=True),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        return io.NodeOutput(text)

class BatchString(io.ComfyNode):
    # NOTE on workflow compatibility: the old V1 node relied on
    # js/batch_condition.js to add "text{n}" STRING widget-inputs
    # client-side beyond what INPUT_TYPES declared, consumed via an
    # unbounded **kwargs pattern (encode() rebuilt the list from
    # kwargs["text1"], kwargs["text2"], ...). This migrates to the official
    # V3 Autogrow dynamic-input API with explicit names "text1".."textN" so
    # the resolved kwarg names match the old JS-generated names exactly.
    # Old workflows that used up to MAX_TEXTS inputs should therefore
    # reconnect by name.
    MAX_TEXTS = 50

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.Autogrow.TemplateNames(
            input=io.String.Input("text", multiline=True),
            names=[f"text{i}" for i in range(1, cls.MAX_TEXTS + 1)],
            min=0,
        )
        return io.Schema(
            node_id=f"BatchString{NODE_SURFIX}",
            display_name=f"Batch String {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Autogrow.Input("texts", template=template),
            ],
            outputs=[
                io.Custom("BATCH_STRING").Output(),
            ],
        )

    @classmethod
    def execute(cls, texts: io.Autogrow.Type) -> io.NodeOutput:
        return io.NodeOutput(list(texts.values()))

class PrefixString(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"PrefixString{NODE_SURFIX}",
            display_name=f"Prefix String {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.String.Input("prefix", multiline=True),
                io.Custom("BATCH_STRING").Input("prompts"),
            ],
            outputs=[
                io.Custom("BATCH_STRING").Output(),
            ],
        )

    @classmethod
    def execute(cls, prefix, prompts) -> io.NodeOutput:
        return io.NodeOutput([prefix + prompt for prompt in prompts])

class SaveBatchString(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"SaveBatchString{NODE_SURFIX}",
            display_name=f"Save Batch String {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("BATCH_STRING").Input("prompts"),
                io.String.Input("folder", default=""),
                io.String.Input("extension", default="txt"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, prompts, folder, extension, seed) -> io.NodeOutput:
        os.makedirs(os.path.join(CURRENT_DIR, folder), exist_ok=True)
        for i, prompt in enumerate(prompts):
            path = os.path.join(CURRENT_DIR, folder, f"{seed:06}_{i:03}.{extension}")
            with open(path, "w") as f:
                f.write(prompt)
        return io.NodeOutput()

class SaveImageBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"SaveImageBatch{NODE_SURFIX}",
            display_name=f"Save Image Batch {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("folder", default=""),
                io.String.Input("extension", default="png"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, folder, extension, seed) -> io.NodeOutput:
        os.makedirs(os.path.join(CURRENT_DIR, folder), exist_ok=True)
        for i, image in enumerate(images):
            path = os.path.join(CURRENT_DIR, folder, f"{seed:06}_{i:03}.{extension}")
            Image.fromarray((image.float().cpu() * 255).numpy().astype('uint8')).save(path)
        return io.NodeOutput()

class SaveLatentBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"SaveLatentBatch{NODE_SURFIX}",
            display_name=f"Save Latent Batch {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Latent.Input("latents"),
                io.String.Input("folder", default=""),
                io.Combo.Input("extension", options=["npy", "npz"], default="npy"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latents, folder, extension, seed) -> io.NodeOutput:
        os.makedirs(os.path.join(CURRENT_DIR, folder), exist_ok=True)
        for i, latent in enumerate(latents["samples"]):
            path = os.path.join(CURRENT_DIR, folder, f"{seed:06}_{i:03}.{extension}")
            if extension == "npy":
                np.save(path, latent.float().cpu().numpy())
            else:
                original_size = (latent.shape[1] * 8, latent.shape[2] * 8)
                crop_ltrb = (0, 0, 0, 0)
                np.savez(
                    path,
                    latents=latent.float().cpu().numpy(),
                    original_size=np.array(original_size),
                    crop_ltrb=np.array(crop_ltrb),
                )
        return io.NodeOutput()

class RandomColorPrompt(io.ComfyNode):
    MAGIC_WORD = "<color>"
    COLORS = [
        "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown",
        "black", "white", "gray", "aqua",
    ]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"RandomColorPrompt{NODE_SURFIX}",
            display_name=f"Random Color Prompt {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.String.Input("base_prompt", default="", multiline=True),
                io.Int.Input("num_prompts", default=4, min=1, max=100),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[
                io.Custom("BATCH_STRING").Output(),
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, base_prompt, num_prompts, seed) -> io.NodeOutput:
        rng = np.random.RandomState(seed)
        prompts = []
        for _ in range(num_prompts):
            prompt = base_prompt
            while cls.MAGIC_WORD in prompt:
                color = rng.choice(cls.COLORS)
                prompt = prompt.replace(cls.MAGIC_WORD, color, 1)
            prompts.append(prompt)

        return_string = "\n\n".join(prompts)
        return io.NodeOutput(prompts, return_string)
