from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
from transformers.generation.logits_process import UnbatchedClassifierFreeGuidanceLogitsProcessor
import comfy
import torch
import re
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, NODE_SURFIX, SYMBOL

CATEGORY_NAME = ROOT_NAME + "dart"

class LoadDart(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"LoadDart{NODE_SURFIX}",
            display_name=f"Load Dart {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.String.Input("tokenizer", default="p1atdev/dart-v1-sft"),
                io.String.Input("model", default="p1atdev/dart-v1-sft"),
            ],
            outputs=[
                io.Custom("DART_TOKENIZER").Output(),
                io.Custom("DART_MODEL").Output(),
            ],
        )

    @classmethod
    def execute(cls, tokenizer, model) -> io.NodeOutput:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        return io.NodeOutput(tokenizer, model)

class DartPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"DartPrompt{NODE_SURFIX}",
            display_name=f"Dart Prompt {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Combo.Input("rating", options=["general", "sensitive", "questionable", "explicit", "sfw", "nsfw"]),
                io.String.Input("copyright", default="original"),
                io.String.Input("character", default=""),
                io.String.Input("general", multiline=True),
                io.Combo.Input("long", options=["very_short", "short", "long", "very_long"], default="long"),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, rating, copyright, character, general, long) -> io.NodeOutput:
        prompt = "<|bos|>"
        prompt += f"<rating>rating:{rating}</rating>"
        prompt += f"<copylight>{copyright}</copyright>"
        prompt += f"<character>{character}</character>"
        prompt += "<general>" + f"<|{long}|>"
        prompt += f"{general}"
        prompt += "<|input_end|>"

        return io.NodeOutput(prompt)

class DartPromptV2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"DartPromptV2{NODE_SURFIX}",
            display_name=f"Dart Prompt V2 {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Combo.Input("rating", options=["general", "sensitive", "questionable", "explicit", "sfw", "nsfw"]),
                io.String.Input("copyright", default="original"),
                io.String.Input("character", default=""),
                io.String.Input("general", multiline=True),
                io.Combo.Input("aspect_ratio", options=["ultra_wide", "wide", "square", "tall", "ultra_tall"], default="tall"),
                io.Combo.Input("length", options=["very_short", "short", "medium", "long", "very_long"], default="medium"),
                io.Combo.Input("identity", options=["none", "lax", "strict"], default="none"),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, rating, copyright, character, general, aspect_ratio, length, identity) -> io.NodeOutput:
        prompt = "<|bos|>"
        prompt += f"<copylight>{copyright}</copyright>"
        prompt += f"<character>{character}</character>"
        prompt += f"<|rating:{rating}|>" + f"<|aspect_ratio:{aspect_ratio}|>" + f"<|length:{length}|>" + f"<|identity:{identity}|>"
        prompt += f"<general>{general}<|identity:{identity}|><|input_end|>"

        return io.NodeOutput(prompt)

class DartConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"DartConfig{NODE_SURFIX}",
            display_name=f"Dart Config {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Int.Input("max_new_tokens", default=128, min=1, max=256, step=1),
                io.Int.Input("min_new_tokens", default=0, min=0, max=255, step=1),
                io.Float.Input("temperature", default=1.0, min=0.0, max=5.0, step=0.01),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=20, min=1, max=500, step=1),
                io.Int.Input("num_beams", default=1, min=1, max=10, step=1),
                io.Float.Input("cfg_scale", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Custom("DART_CONFIG").Output(),
            ],
        )

    @classmethod
    def execute(cls, max_new_tokens, min_new_tokens, temperature, top_p, top_k, num_beams, cfg_scale) -> io.NodeOutput:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_beams": num_beams,
            "cfg_scale": cfg_scale,
        }
        kwargs["temperature"] = float(kwargs["temperature"]) # avoid error
        return io.NodeOutput(kwargs)

class BanTags(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"BanTags{NODE_SURFIX}",
            display_name=f"Ban Tags {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("DART_TOKENIZER").Input("tokenizer"),
                io.String.Input("ban_tags", multiline=True),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, tokenizer, ban_tags) -> io.NodeOutput:
        ban_tags_result = set()
        patterns = [re.compile(ban_tag) for ban_tag in ban_tags.splitlines()]
        for pattern in patterns:
            for tag in tokenizer.vocab:
                if pattern.match(tag):
                    ban_tags_result.add(tag)
        return io.NodeOutput(", ".join(ban_tags_result))

class DartGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"DartGenerate{NODE_SURFIX}",
            display_name=f"Dart Generate {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Custom("DART_TOKENIZER").Input("tokenizer"),
                io.Custom("DART_MODEL").Input("model"),
                io.String.Input("prompt", default=""),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Custom("DART_CONFIG").Input("config", optional=True),
                io.String.Input("negative", default="", optional=True),
                io.String.Input("ban_tags", default="", optional=True),
            ],
            outputs=[
                io.Custom("BATCH_STRING").Output(),
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, tokenizer, model, prompt, batch_size, seed, config=None, negative=None, ban_tags=None) -> io.NodeOutput:
        if config:
            config = config
        else:
            config = {
                "max_new_tokens": 128,
                "min_new_tokens": 0,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 100,
                "num_beams": 1,
            }

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        generation_config = GenerationConfig.from_pretrained("p1atdev/dart-v1-sft", **config) # こんなんでいいの？
        model.to(comfy.model_management.get_torch_device(), dtype=torch.float16).eval()
        inputs = tokenizer([prompt], return_tensors="pt").input_ids.to(comfy.model_management.get_torch_device()).repeat(batch_size, 1)
        if config["cfg_scale"] != 1.0:
            negative_inputs = tokenizer([negative], return_tensors="pt").input_ids.to(comfy.model_management.get_torch_device()).repeat(batch_size, 1)
            loggits_processor = LogitsProcessorList([
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    guidance_scale=config["cfg_scale"],
                    model=model,
                    unconditional_ids=negative_inputs,
                )
            ])
        else:
            loggits_processor = None

        if ban_tags:
            ban_tags_ids = tokenizer([ban_tags]).input_ids
            bad_words_ids = [[token_id] for token_id in ban_tags_ids[0]]
        else:
            bad_words_ids = None

        with torch.no_grad():
            outputs = model.generate(inputs, generation_config=generation_config, bad_words_ids=bad_words_ids, logits_processor=loggits_processor)

        prompts = [", ".join([tag for tag in tokenizer.batch_decode(output, skip_special_tokens=True) if tag.strip() != ""]) for output in outputs]

        if "rating:" in prompts[0]:
            # delete rating
            prompts = [", ".join(prompt.split(", ")[1:]) for prompt in prompts]

        strings = "" # for checking
        for i, prompt in enumerate(prompts):
            strings += f"Prompt {i + 1}:\n{prompt}\n\n"

        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

        return io.NodeOutput(prompts, strings)
