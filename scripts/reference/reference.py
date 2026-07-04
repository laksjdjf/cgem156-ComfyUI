import torch
from comfy_api.v0_0_2 import io
from ... import ROOT_NAME, SYMBOL, NODE_SURFIX

CATEGORY_NAME = ROOT_NAME + "reference"

class ReferenceApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"ReferenceApply{NODE_SURFIX}",
            display_name=f"Reference Apply {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("index", default=0, min=0, max=256),
                io.Combo.Input("mode", options=["concat", "replace"], default="concat"),
                io.Int.Input("depth", default=12, min=-1, max=12),
                io.Float.Input("start_step", default=0, min=0, max=1, step=0.01),
                io.Float.Input("end_step", default=1, min=0, max=1, step=0.01),
                io.Boolean.Input("apply_input", default=True),
                io.Boolean.Input("apply_middle", default=True),
                io.Boolean.Input("apply_output", default=True),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, index, mode, depth, start_step, end_step, apply_input, apply_middle, apply_output) -> io.NodeOutput:
        model_reference = model.clone()
        start_sigma = model_reference.model.model_sampling.percent_to_sigma(start_step)
        end_sigma = model_reference.model.model_sampling.percent_to_sigma(end_step)

        sdxl = hasattr(model_reference.model.diffusion_model, "label_emb")
        num_blocks = 8 if sdxl else 11

        def reference_apply(q, k, v, extra_options):
            block_name, block_id = extra_options["block"]
            chunks = len(extra_options["cond_or_uncond"])
            batch_size = q.shape[0] // chunks

            if block_name == "input" and not apply_input:
                return q, k, v
            if block_name == "middle" and not apply_middle:
                return q, k, v
            if block_name == "output" and not apply_output:
                return q, k, v

            if block_name == "output":
                block_number = num_blocks - block_id
            else:
                block_number = block_id

            q_out = q.clone()
            k_out = k.clone()
            v_out = v.clone()

            sigma = extra_options["sigmas"][0].item()

            if end_sigma <= sigma <= start_sigma and block_number <= depth:
                k_ref = k_out[index::batch_size].repeat_interleave(batch_size, dim=0).clone()
                v_ref = v_out[index::batch_size].repeat_interleave(batch_size, dim=0).clone()

                k_out = torch.cat([k_out, k_ref], dim=1) if mode == "concat" else k_ref
                v_out = torch.cat([v_out, v_ref], dim=1) if mode == "concat" else v_ref

            return q_out, k_out, v_out

        model_reference.set_model_attn1_patch(reference_apply)

        return io.NodeOutput(model_reference)

class ReferenceLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"ReferenceLatent{NODE_SURFIX}",
            display_name=f"Reference Latent {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Latent.Input("latent"),
                io.Int.Input("index", default=0, min=0, max=256),
                io.Int.Input("batch_size", default=1, min=1, max=256),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, latent, index, batch_size) -> io.NodeOutput:
        latent_new = latent.copy()

        sample = latent_new["samples"]
        height, width = sample.shape[2], sample.shape[3]

        empty_latent = torch.zeros_like(latent["samples"]).repeat(batch_size , 1, 1, 1)
        empty_latent[index] = sample[0]
        noise_mask = torch.ones(batch_size, 1, height * 8, width * 8).to(sample)
        noise_mask[index] = 0.0

        latent_new["samples"] = empty_latent
        latent_new["noise_mask"] = noise_mask

        return io.NodeOutput(latent_new)

class MultipleReferenceApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"MultipleReferenceApply{NODE_SURFIX}",
            display_name=f"Multiple Reference Apply {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Model.Input("model"),
                io.String.Input("indices", default="0"),
                io.Int.Input("depth", default=12, min=-1, max=12),
                io.Float.Input("start_step", default=0, min=0, max=1, step=0.01),
                io.Float.Input("end_step", default=1, min=0, max=1, step=0.01),
                io.Boolean.Input("apply_input", default=True),
                io.Boolean.Input("apply_middle", default=True),
                io.Boolean.Input("apply_output", default=True),
                io.String.Input("weights", default=""),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, indices, depth, start_step, end_step, apply_input, apply_middle, apply_output, weights) -> io.NodeOutput:
        model_reference = model.clone()
        start_sigma = model_reference.model.model_sampling.percent_to_sigma(start_step)
        end_sigma = model_reference.model.model_sampling.percent_to_sigma(end_step)

        sdxl = hasattr(model_reference.model.diffusion_model, "label_emb")
        num_blocks = 8 if sdxl else 11

        indices = [int(i) for i in indices.split(",") if i.strip().isdigit()]
        weights = [float(i) for i in weights.split(",") if i.strip()] if weights else [1.0] * len(indices)

        def reference_apply(q, k, v, extra_options):
            block_name, block_id = extra_options["block"]


            if block_name == "input" and not apply_input:
                return q, k, v
            if block_name == "middle" and not apply_middle:
                return q, k, v
            if block_name == "output" and not apply_output:
                return q, k, v

            if block_name == "output":
                block_number = num_blocks - block_id
            else:
                block_number = block_id

            q_out = q.clone()
            k_out = k.clone()
            v_out = v.clone()

            sigma = extra_options["sigmas"][0].item()


            if end_sigma <= sigma <= start_sigma and block_number <= depth:
                chunks = len(extra_options["cond_or_uncond"])
                batch_size = q.shape[0] // chunks
                num_tokens = q.shape[1]

                k_refs = torch.cat([k_out[i::batch_size] for i in indices], dim=1)
                v_refs = torch.cat([v_out[i::batch_size] * weight for i, weight in zip(indices, weights)], dim=1)

                k_out = k_out.repeat(1, len(indices)+1, 1).clone()
                v_out = v_out.repeat(1, len(indices)+1, 1).clone()
                for i in range(batch_size):
                    if i not in indices:
                        k_out[i::batch_size, num_tokens:] = k_refs.clone()
                        v_out[i::batch_size, num_tokens:] = v_refs.clone()

            return q_out, k_out, v_out

        model_reference.set_model_attn1_patch(reference_apply)

        return io.NodeOutput(model_reference)

class MultipleReferenceLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"MultipleReferenceLatent{NODE_SURFIX}",
            display_name=f"Multiple Reference Latent {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Latent.Input("latent"),
                io.String.Input("indices", default="0"),
                io.Int.Input("batch_size", default=1, min=1, max=256),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, latent, indices, batch_size) -> io.NodeOutput:
        latent_new = latent.copy()
        indices = [int(i) for i in indices.split(",") if i.strip().isdigit()]

        sample = latent_new["samples"]
        b, _, height, width = sample.shape

        assert len(indices) == b

        empty_latent = torch.zeros_like(latent["samples"][:1]).repeat(batch_size , 1, 1, 1)
        empty_latent[torch.tensor(indices)] = sample
        noise_mask = torch.ones(batch_size, 1, height * 8, width * 8).to(sample)
        noise_mask[torch.tensor(indices)] = 0.0

        latent_new["samples"] = empty_latent
        latent_new["noise_mask"] = noise_mask

        return io.NodeOutput(latent_new)
