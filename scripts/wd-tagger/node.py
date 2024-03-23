from .preprocess import preprocess
import timm
import pandas as pd
import torch
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
    
    RETURN_TYPES = ("BATCH_STRING", "STRING")
    FUNCTION = "predict_tag"
    CATEGORY = CATEGORY_NAME

    def predict_tag(self, tagger, labels, image, rating, character_thereshold, general_thereshold):
        dtype = tagger.parameters().__next__().dtype
        image = preprocess(image).to("cuda", dtype=dtype)
        with torch.no_grad():
            logits = tagger(image)
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
        return (prompts, string)