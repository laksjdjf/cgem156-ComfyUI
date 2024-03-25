from .preprocess import preprocess
import timm
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ... import ROOT_NAME

CATEGORY_NAME = ROOT_NAME + "wd-tagger"

MODEL_REPO_MAP = [
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
]

class LoadTagger:
    def __init__(self):
        self.loaded_model = None
        self.loaded_df = None
        self.loaded_model_name = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "tagger": (MODEL_REPO_MAP,),
                "dtype": (["fp16", "fp32", "bf16"], ),
            }
        }
    RETURN_TYPES = ("WD_TAGGER", "WD_TAGGER_LABELS")
    FUNCTION = "load_tagger"

    CATEGORY = CATEGORY_NAME

    @torch.inference_mode(False)
    def load_tagger(self, tagger, dtype):
        
        if self.loaded_model_name != tagger:
            self.loaded_model_name = tagger
            self.loaded_model = timm.create_model(f"hf_hub:{tagger}", pretrained=True)
            self.loaded_df = pd.read_csv(f"https://huggingface.co/{tagger}/resolve/main/selected_tags.csv")
        self.dtype = torch.float16 if dtype == "fp16" else torch.float32 if dtype == "fp32" else torch.bfloat16
        self.loaded_model = self.loaded_model.to("cuda", dtype=self.dtype).eval()

        return (self.loaded_model, self.loaded_df)
    
class PredictTag:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "tagger": ("WD_TAGGER",),
                "labels": ("WD_TAGGER_LABELS",),
                "image": ("IMAGE",),
                "rating": ("BOOLEAN", {"default": False}),
                "character_thereshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.001, "step": 0.001}),
                "general_thereshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.001, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("BATCH_STRING", "STRING", "WD-TAGGER-FEATURES")
    FUNCTION = "predict_tag"
    CATEGORY = CATEGORY_NAME

    @torch.inference_mode(False)
    def predict_tag(self, tagger, labels, image, rating, character_thereshold, general_thereshold):
        dtype = tagger.parameters().__next__().dtype
        preprocessed_image = preprocess(image).to("cuda", dtype=dtype)
        with torch.no_grad():
            feature = tagger.forward_features(preprocessed_image)
            logits = tagger.forward_head(feature)
            probs = logits.sigmoid()
        probs = probs.cpu().numpy()

        prompts = []
        for prob in probs:
            labels["prob"] = prob
            sorted_labels = labels.sort_values(by="prob", ascending=False)
            tags = []
            if rating:
                tags.append(sorted_labels[sorted_labels["category"] == 9]["name"].to_list()[0])
            character_tags = sorted_labels[(sorted_labels["prob"] > character_thereshold) & (sorted_labels["category"] == 4)]["name"].to_list()
            general_tags = sorted_labels[(sorted_labels["prob"] > general_thereshold) & (sorted_labels["category"] == 0)]["name"].to_list()
            
            tags += character_tags + general_tags
            prompt = ", ".join([tag.replace("_", " ") for tag in tags])
            prompts.append(prompt)

        string = "\n".join([f"prompt:{i}\n{prompt}" for i, prompt in enumerate(prompts)])
        id_to_tag = labels['name'].to_dict()
        tag_to_id = {v:k for k,v in id_to_tag.items()}
        
        features = {
            "feature": feature,
            "image": ((preprocessed_image + 1) / 2).flip(1).permute(0, 2, 3, 1).float().cpu(),  # なにこれは・・・
            "tag_to_id": tag_to_id,
        }
        
        return (prompts, string, features)
    
class GradCam:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "tagger": ("WD_TAGGER",),
                "features": ("WD-TAGGER-FEATURES",),
                "target_tag": ("STRING",{"default": ""}),
                "heat_map_alpha": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "grad_cam"
    CATEGORY = CATEGORY_NAME

    @torch.inference_mode(False)
    def grad_cam(self, tagger, features, target_tag, heat_map_alpha):
        
        image = features["image"]
        size = (image.shape[1], image.shape[2])
        target_id = features["tag_to_id"][target_tag.strip().replace(" ", "_")]

        features = features["feature"].detach().clone().requires_grad_(True)
        
        gradients = []
        for i in range(len(features)):
            feature = features[i].unsqueeze(0)
            outputs = tagger.forward_head(feature).sigmoid()
            output = outputs[:,target_id]
            gradients.append(torch.autograd.grad(output, feature, retain_graph=True)[0])
            tagger.zero_grad()
            features.grad = None
        gradients = torch.cat(gradients)

        weight = torch.mean(gradients, dim=1, keepdim=True)
        heat_map = torch.sum(weight * features, dim=2).relu().reshape(-1, 1, 28, 28)
        heat_map = heat_map / heat_map.max()

        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = heat_map.reshape(-1, 1).detach().float().cpu().numpy()

        c_map = plt.get_cmap("jet")
        heat_map = c_map(heat_map).reshape(-1, 28, 28, 4)[:,:,:,:3]
        heat_map = torch.from_numpy(heat_map)
        heat_map = heat_map.permute(0, 3, 1, 2)
        heat_map = torch.nn.functional.interpolate(heat_map, size=size, mode="bilinear", align_corners=False)
        heat_map = heat_map.permute(0, 2, 3, 1)

        return (image * (1 - heat_map_alpha) + heat_map * heat_map_alpha, )