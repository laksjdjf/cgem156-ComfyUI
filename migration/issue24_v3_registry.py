"""Issue #24: ComfyUI V3 migration registry for cgem156 custom nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

Difficulty = Literal["A", "B", "C"]


@dataclass(frozen=True)
class NodeMigrationRecord:
    node_name: str
    file_path: str
    difficulty: Difficulty
    v1_components: tuple[str, ...]
    migration_notes: tuple[str, ...]


NODE_MIGRATION_RECORDS: tuple[NodeMigrationRecord, ...] = (
    NodeMigrationRecord("CLIPTextEncodeBatch", "scripts/batch_condition/node.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom type: BATCH_STRING", "variable length conditioning merge")),
    NodeMigrationRecord("StringInput", "scripts/batch_condition/node.py", "A", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ()),
    NodeMigrationRecord("BatchString", "scripts/batch_condition/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("dynamic inputs via **kwargs", "JavaScript add/remove inputs integration")),
    NodeMigrationRecord("PrefixString", "scripts/batch_condition/node.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom type: BATCH_STRING",)),
    NodeMigrationRecord("SaveBatchString", "scripts/batch_condition/node.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY", "OUTPUT_NODE"), ("filesystem side effects",)),
    NodeMigrationRecord("SaveImageBatch", "scripts/batch_condition/node.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY", "OUTPUT_NODE"), ("filesystem side effects",)),
    NodeMigrationRecord("SaveLatentBatch", "scripts/batch_condition/node.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY", "OUTPUT_NODE"), ("filesystem side effects",)),
    NodeMigrationRecord("AttentionCouple", "scripts/attention_couple/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("dynamic cond/mask inputs", "model attention patching", "stateful attributes")),
    NodeMigrationRecord("LortnocLoader", "scripts/lortnoc/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("stateful cache in __init__", "model input block patching")),
    NodeMigrationRecord("CDTuner", "scripts/cd_tuner/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("UNet weight rewrite wrapper", "stateful temporary tensors")),
    NodeMigrationRecord("LoraLoaderFromWeight", "scripts/lora_merger/load.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom type: LoRA",)),
    NodeMigrationRecord("LoraLoaderWeightOnly", "scripts/lora_merger/load.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("stateful cache in __init__", "LoRA key transformation", "custom type: LoRA")),
    NodeMigrationRecord("LoraMerger", "scripts/lora_merger/merge.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("SVD/concat/add merge modes", "custom type: LoRA")),
    NodeMigrationRecord("LoraSVDRank", "scripts/lora_merger/merge.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("SVD rank estimation", "custom type: LoRA")),
    NodeMigrationRecord("LoraSave", "scripts/lora_merger/save.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY", "OUTPUT_NODE"), ("filesystem side effects", "custom type: LoRA")),
    NodeMigrationRecord("MultipleLoraLoader3", "scripts/multiple_lora_loader/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("dynamic class generation", "stateful cache in __init__", "variable repeated inputs")),
    NodeMigrationRecord("MultipleLoraLoader5", "scripts/multiple_lora_loader/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("dynamic class generation", "stateful cache in __init__", "variable repeated inputs")),
    NodeMigrationRecord("MultipleLoraLoader10", "scripts/multiple_lora_loader/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("dynamic class generation", "stateful cache in __init__", "variable repeated inputs")),
    NodeMigrationRecord("GradualLatentSampler", "scripts/custom_samplers/gradual_latent.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom sampler behavior", "many sampler parameters")),
    NodeMigrationRecord("LCMSamplerRCFG", "scripts/custom_samplers/lcm_sampler_rcfg.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom sampler", "optional latent input branch")),
    NodeMigrationRecord("TCDSampler", "scripts/custom_samplers/tcd_sampler.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom sampler",)),
    NodeMigrationRecord("SamplerCustomAdvancedPreview", "scripts/custom_samplers/sampler_custom_preview.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY"), ("preview callback integration", "extra output image stream")),
    NodeMigrationRecord("SamplerEulerAncestralFixedNoise", "scripts/custom_samplers/euler_ancestral_fixed_noise.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("noise sampler selection",)),
    NodeMigrationRecord("TextScheduler", "scripts/custom_schedulers/text_scheduler.py", "A", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ()),
    NodeMigrationRecord("LimitedIntervalCFGGuider", "scripts/custom_guiders/limited_interval_cfg_guider.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom GUIDER object",)),
    NodeMigrationRecord("VariationNoise", "scripts/custom_noise/variation_noise.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom NOISE generator object",)),
    NodeMigrationRecord("RandomNoiseOffset", "scripts/custom_noise/variation_noise.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom NOISE generator object",)),
    NodeMigrationRecord("RandomNoiseVariationSimple", "scripts/custom_noise/variation_noise.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom NOISE generator object",)),
    NodeMigrationRecord("ScaleCrafter", "scripts/scale_crafter/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("runtime Conv2d forward patching",)),
    NodeMigrationRecord("LoadAestheticShadow", "scripts/aesthetic_shadow/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("external model pipeline loading", "optional attention forward replacement")),
    NodeMigrationRecord("PredictAesthetic", "scripts/aesthetic_shadow/node.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom model type: AESTHETIC_SHADOW_MODEL",)),
    NodeMigrationRecord("AttentionScale", "scripts/for_test/attention_scale.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("attention replacement logic",)),
    NodeMigrationRecord("LoraLoaderModelOnlyXY", "scripts/lora_xy/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom types: XY_MODEL/XY_LIST",)),
    NodeMigrationRecord("SamplerCustomXY", "scripts/lora_xy/node.py", "C", ("INPUT_TYPES", "FUNCTION", "CATEGORY"), ("inherits return metadata", "model list iteration")),
    NodeMigrationRecord("KSamplerXY", "scripts/lora_xy/node.py", "C", ("INPUT_TYPES", "FUNCTION", "CATEGORY"), ("inherits return metadata", "model list iteration")),
    NodeMigrationRecord("KSamplerAdvancedXY", "scripts/lora_xy/node.py", "C", ("INPUT_TYPES", "FUNCTION", "CATEGORY"), ("inherits return metadata", "model list iteration")),
    NodeMigrationRecord("PreviewXY", "scripts/lora_xy/node.py", "C", ("INPUT_TYPES",), ("hidden inputs (PROMPT/EXTRA_PNGINFO)", "UI dictionary output", "image save side effects")),
    NodeMigrationRecord("LoadDart", "scripts/dart/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("external tokenizer/model loading", "custom types: DART_TOKENIZER/DART_MODEL")),
    NodeMigrationRecord("DartPrompt", "scripts/dart/node.py", "A", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ()),
    NodeMigrationRecord("DartPromptV2", "scripts/dart/node.py", "A", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ()),
    NodeMigrationRecord("DartConfig", "scripts/dart/node.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom type: DART_CONFIG",)),
    NodeMigrationRecord("BanTags", "scripts/dart/node.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("tokenizer vocab dependent matching",)),
    NodeMigrationRecord("DartGenerate", "scripts/dart/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("generation config flow", "RNG state save/restore", "custom types")),
    NodeMigrationRecord("ReferenceApply", "scripts/reference/reference.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("attention patch hook", "sigma/depth gating")),
    NodeMigrationRecord("ReferenceLatent", "scripts/reference/reference.py", "B", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("latent/noise_mask transformation",)),
    NodeMigrationRecord("LoadTagger", "scripts/wd-tagger/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("external model download", "stateful cache in __init__", "CUDA/dtype management")),
    NodeMigrationRecord("PredictTag", "scripts/wd-tagger/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("custom feature bundle: WD-TAGGER-FEATURES",)),
    NodeMigrationRecord("GradCam", "scripts/wd-tagger/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("autograd heatmap flow",)),
    NodeMigrationRecord("GradCamAuto", "scripts/wd-tagger/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("autograd + dynamic tag selection",)),
    NodeMigrationRecord("GradPair", "scripts/wd-tagger/node.py", "C", ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"), ("pairwise feature comparison + autograd",)),
)


P0_PRIORITY_TOPICS: tuple[str, ...] = (
    "PoC dynamic input + JS linked nodes (AttentionCouple, BatchString, PreviewXY hidden inputs)",
    "Define V3 caching strategy for stateful nodes (__init__/self.loaded_* patterns)",
    "Confirm V3 equivalents for model patch APIs (set_model_attn*, set_model_unet_function_wrapper, set_model_input_block_patch, KSAMPLER)",
)

P1_PRIORITY_TOPICS: tuple[str, ...] = (
    "Migrate C-tier nodes by cluster: sampler/guider/noise",
    "Migrate C-tier nodes by cluster: model patching",
    "Migrate C-tier nodes by cluster: external model integrations (Dart/wd-tagger/aesthetic-shadow)",
)

P2_PRIORITY_TOPICS: tuple[str, ...] = (
    "Batch migrate B-tier custom-type nodes",
    "Stabilize OUTPUT_NODE side-effect boundaries with explicit regression checks",
)

P3_PRIORITY_TOPICS: tuple[str, ...] = (
    "Finish A-tier syntax-level migrations",
    "Consolidate NODE_CLASS_MAPPINGS/NODE_DISPLAY_NAME_MAPPINGS for final V3 schema",
)

ACCEPTANCE_CHECKLIST: tuple[str, ...] = (
    "Input/output type compatibility validated (including custom types)",
    "Hidden input and JS/UI integration compatibility validated",
    "Output node side effects validated (save paths/content)",
    "Patch/sampler behavior remains within acceptable output tolerance",
    "State/cache dependent nodes re-execution behavior validated",
)


def iter_records_by_difficulty(level: Difficulty) -> Iterable[NodeMigrationRecord]:
    return (record for record in NODE_MIGRATION_RECORDS if record.difficulty == level)


def count_by_difficulty() -> dict[Difficulty, int]:
    return {
        "A": sum(1 for _ in iter_records_by_difficulty("A")),
        "B": sum(1 for _ in iter_records_by_difficulty("B")),
        "C": sum(1 for _ in iter_records_by_difficulty("C")),
    }
