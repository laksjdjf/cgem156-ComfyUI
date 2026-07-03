import comfy
import torch
from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "for_test"

def reset_weight(tokens):
    ret_dic = {}
    for key in tokens:
        ret_dic[key] = [[(token, 1) for token, weight in tokens[key][0]]]
        weights = [weight for token, weight in tokens[key][0]]
    return ret_dic, weights

class CLIPTextEncodeBatchKVMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ), 
                "clip": ("CLIP", ), 
                "text_k":("STRING", {"multiline": True}),
                "text_v":("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, model, clip, text_k, text_v):
        
        tokens_k = clip.tokenize(text_k)
        tokens_v = clip.tokenize(text_v)

        tokens_no_weight_k, k_weights = reset_weight(tokens_k)
        tokens_no_weight_v, v_weights = reset_weight(tokens_v)

        assert tokens_no_weight_k == tokens_no_weight_v, "tokens_k and tokens_v must be the same."
        cond, pooled = clip.encode_from_tokens(tokens_no_weight_k, return_pooled=True)
        
        self.k_weights = torch.tensor(k_weights).view(1, -1, 1)
        self.v_weights = torch.tensor(v_weights).view(1, -1, 1)

        new_model = model.clone()
        def attn2_patch(q, k, v, extra_options):
            
            assert k.mean() == v.mean(), "k and v must be the same."
            if k.shape[1] != self.k_weights.shape[1]:
                self.k_weights.repeat(1, k.shape[1] // self.k_weights.shape[1], 1)
                self.v_weights.repeat(1, v.shape[1] // self.v_weights.shape[1], 1)

            if self.k_weights.device != k.device:
                self.k_weights = self.k_weights.to(k)
                self.v_weights = self.v_weights.to(v)

            ks = k * self.k_weights
            vs = v * self.v_weights

            return q, ks, vs

        new_model.set_model_attn2_patch(attn2_patch)

        return (new_model, [[cond, {"pooled_output": pooled}]])

