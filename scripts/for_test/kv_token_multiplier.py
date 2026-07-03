import comfy
import torch
from ... import ROOT_NAME, SYMBOL, NODE_SURFIX
from comfy_api.v0_0_2 import io

CATEGORY_NAME = ROOT_NAME + "for_test"

def reset_weight(tokens):
    ret_dic = {}
    for key in tokens:
        ret_dic[key] = [[(token, 1) for token, weight in tokens[key][0]]]
        weights = [weight for token, weight in tokens[key][0]]
    return ret_dic, weights

class CLIPTextEncodeBatchKVMultiply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"CLIPTextEncodeBatchKVMultiply{NODE_SURFIX}",
            display_name=f"CLIP Text Encode Batch KV Multiply {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("text_k", multiline=True),
                io.String.Input("text_v", multiline=True),
            ],
            outputs=[
                io.Model.Output(),
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, clip, text_k, text_v) -> io.NodeOutput:

        tokens_k = clip.tokenize(text_k)
        tokens_v = clip.tokenize(text_v)

        tokens_no_weight_k, k_weights = reset_weight(tokens_k)
        tokens_no_weight_v, v_weights = reset_weight(tokens_v)

        assert tokens_no_weight_k == tokens_no_weight_v, "tokens_k and tokens_v must be the same."
        cond, pooled = clip.encode_from_tokens(tokens_no_weight_k, return_pooled=True)

        state = {
            "k_weights": torch.tensor(k_weights).view(1, -1, 1),
            "v_weights": torch.tensor(v_weights).view(1, -1, 1),
        }

        new_model = model.clone()
        def attn2_patch(q, k, v, extra_options):

            assert k.mean() == v.mean(), "k and v must be the same."
            if k.shape[1] != state["k_weights"].shape[1]:
                state["k_weights"].repeat(1, k.shape[1] // state["k_weights"].shape[1], 1)
                state["v_weights"].repeat(1, v.shape[1] // state["v_weights"].shape[1], 1)

            if state["k_weights"].device != k.device:
                state["k_weights"] = state["k_weights"].to(k)
                state["v_weights"] = state["v_weights"].to(v)

            ks = k * state["k_weights"]
            vs = v * state["v_weights"]

            return q, ks, vs

        new_model.set_model_attn2_patch(attn2_patch)

        return io.NodeOutput(new_model, [[cond, {"pooled_output": pooled}]])
