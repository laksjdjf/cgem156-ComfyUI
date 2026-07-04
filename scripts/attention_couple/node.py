import torch
import torch.nn.functional as F
import comfy
import math
from types import SimpleNamespace
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, NODE_SURFIX, SYMBOL

CATEGORY_NAME = ROOT_NAME + "attention_couple"

# Max number of extra cond/mask pairs the UI can grow to via Autogrow.
MAX_PAIRS = 50

def get_mask(mask, batch_size, num_tokens, original_shape):
    num_conds = mask.shape[0]

    if original_shape[2] * original_shape[3] == num_tokens:
        down_sample_rate = 1
    elif (original_shape[2] // 2) * (original_shape[3] // 2) == num_tokens:
        down_sample_rate = 2
    elif (original_shape[2] // 4) * (original_shape[3] // 4) == num_tokens:
        down_sample_rate = 4
    else:
        down_sample_rate = 8

    size = (original_shape[2] // down_sample_rate, original_shape[3] // down_sample_rate)
    mask_downsample = F.interpolate(mask, size=size, mode="nearest")
    mask_downsample = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(batch_size, dim=0)

    return mask_downsample

def lcm(a, b):
    return a * b // math.gcd(a, b)

def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm

class AttentionCouple(io.ComfyNode):
    # NOTE on workflow compatibility: the old V1 node exposed a fixed
    # model/base_mask schema and relied on js/attention_couple.js to add
    # cond_N (CONDITIONING) / mask_N (MASK) input pairs client-side beyond
    # what INPUT_TYPES declared, consumed via an unbounded **kwargs pattern.
    # This migrates to the official V3 Autogrow dynamic-input API using two
    # parallel Autogrow.TemplateNames templates (one for "cond_N", one for
    # "mask_N"), with explicit 1-indexed names so the resolved kwarg names
    # match the old JS-generated names exactly (cond_1/mask_1, cond_2/mask_2,
    # ...). Old workflows that used pairs within MAX_PAIRS should therefore
    # reconnect by name; see the migration report for the caveats (fixed
    # upper bound, and cond_N/mask_N no longer forced to be added/removed as
    # a strict pair by the UI).
    @classmethod
    def define_schema(cls) -> io.Schema:
        cond_template = io.Autogrow.TemplateNames(
            input=io.Conditioning.Input("cond"),
            names=[f"cond_{i}" for i in range(1, MAX_PAIRS + 1)],
            min=0,
        )
        mask_template = io.Autogrow.TemplateNames(
            input=io.Mask.Input("mask"),
            names=[f"mask_{i}" for i in range(1, MAX_PAIRS + 1)],
            min=0,
        )
        return io.Schema(
            node_id=f"AttentionCouple{NODE_SURFIX}",
            display_name=f"Attention Couple {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Model.Input("model"),
                io.Mask.Input("base_mask"),
                io.Autogrow.Input("conds", template=cond_template),
                io.Autogrow.Input("masks", template=mask_template),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, base_mask, conds: io.Autogrow.Type, masks: io.Autogrow.Type) -> io.NodeOutput:
        new_model = model.clone()

        # Unlike the old JS UI (which always added/removed cond_i/mask_i as
        # a pair), the two Autogrow blocks now grow independently, so a
        # workflow could connect cond_i without mask_i (or vice versa).
        # Fail fast with a clear message instead of silently misaligning
        # tensors further down.
        cond_indices = {name.split("_", 1)[1] for name in conds}
        mask_indices = {name.split("_", 1)[1] for name in masks}
        assert cond_indices == mask_indices, (
            f"Mismatched cond_N/mask_N inputs: conds={sorted(conds)}, masks={sorted(masks)}. "
            "Every connected cond_N input must have a matching mask_N input, and vice versa."
        )
        num_conds = len(conds) + 1

        mask = [base_mask] + list(masks.values())
        mask = torch.stack(mask, dim=0)
        assert mask.sum(dim=0).min() > 0, "There are areas that are zero in all masks."

        # execute() is a classmethod (no `self`), so the mutable state that
        # attn2_patch/attn2_output_patch share across repeated calls (device
        # caching, batch_size handoff) lives on this small namespace instead
        # of on a node instance. This is a structural translation only; the
        # attention-patching math below is unchanged from the V1 node.
        state = SimpleNamespace(
            mask=mask / mask.sum(dim=0, keepdim=True),
            conds=[cond[0][0] for cond in conds.values()],
            batch_size=None,
        )
        num_tokens = [cond.shape[1] for cond in state.conds]

        def attn2_patch(q, k, v, extra_options):
            assert k.mean() == v.mean(), "k and v must be the same."
            device, dtype = q.device, q.dtype

            if state.conds[0].device != device:
                state.conds = [cond.to(device, dtype=dtype) for cond in state.conds]
            if state.mask.device != device:
                state.mask = state.mask.to(device, dtype=dtype)

            cond_or_unconds = extra_options["cond_or_uncond"]
            num_chunks = len(cond_or_unconds)
            state.batch_size = q.shape[0] // num_chunks
            q_chunks = q.chunk(num_chunks, dim=0)
            k_chunks = k.chunk(num_chunks, dim=0)
            lcm_tokens = lcm_for_list(num_tokens + [k.shape[1]])
            conds_tensor = torch.cat([cond.repeat(state.batch_size, lcm_tokens // num_tokens[i], 1) for i, cond in enumerate(state.conds)], dim=0)

            qs, ks = [], []
            for i, cond_or_uncond in enumerate(cond_or_unconds):
                k_target = k_chunks[i].repeat(1, lcm_tokens // k.shape[1], 1)
                if cond_or_uncond == 1: # uncond
                    qs.append(q_chunks[i])
                    ks.append(k_target)
                else:
                    qs.append(q_chunks[i].repeat(num_conds, 1, 1))
                    ks.append(torch.cat([k_target, conds_tensor], dim=0))

            qs = torch.cat(qs, dim=0)
            ks = torch.cat(ks, dim=0).to(k)

            return qs, ks, ks

        def attn2_output_patch(out, extra_options):

            cond_or_unconds = extra_options["cond_or_uncond"]
            mask_downsample = get_mask(state.mask, state.batch_size, out.shape[1], extra_options["original_shape"])
            outputs = []
            pos = 0
            for cond_or_uncond in cond_or_unconds:
                if cond_or_uncond == 1: # uncond
                    outputs.append(out[pos:pos + state.batch_size])
                    pos += state.batch_size
                else:
                    masked_output = (out[pos:pos + num_conds * state.batch_size] * mask_downsample).view(num_conds, state.batch_size, out.shape[1], out.shape[2])
                    masked_output = masked_output.sum(dim=0)
                    outputs.append(masked_output)
                    pos += num_conds * state.batch_size
            return torch.cat(outputs, dim=0)

        new_model.set_model_attn2_patch(attn2_patch)
        new_model.set_model_attn2_output_patch(attn2_output_patch)

        return io.NodeOutput(new_model)

