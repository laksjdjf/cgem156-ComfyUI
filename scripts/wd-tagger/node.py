from .preprocess import preprocess
import timm
import numpy as np
import cv2
import pandas as pd
import torch
import matplotlib.pyplot as plt
from comfy_api.v0_0_2 import io

from ... import ROOT_NAME, SYMBOL, NODE_SURFIX

CATEGORY_NAME = ROOT_NAME + "wd-tagger"

WDTagger = io.Custom("WD_TAGGER")
WDTaggerLabels = io.Custom("WD_TAGGER_LABELS")
WDTaggerFeatures = io.Custom("WD-TAGGER-FEATURES")
BatchString = io.Custom("BATCH_STRING")

MODEL_REPO_MAP = [
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-eva02-large-tagger-v3",
]

# module-level cache (V3 nodes execute as classmethods, so instance attributes are not available)
_TAGGER_CACHE = {
    "loaded_model": None,
    "loaded_df": None,
    "loaded_model_name": None,
}

class LoadTagger(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"LoadTagger{NODE_SURFIX}",
            display_name=f"Load Tagger {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                io.Combo.Input("tagger", options=MODEL_REPO_MAP),
                io.Combo.Input("dtype", options=["fp16", "fp32", "bf16"]),
            ],
            outputs=[
                WDTagger.Output(),
                WDTaggerLabels.Output(),
            ],
        )

    @classmethod
    @torch.inference_mode(False)
    def execute(cls, tagger, dtype) -> io.NodeOutput:

        if _TAGGER_CACHE["loaded_model_name"] != tagger:
            _TAGGER_CACHE["loaded_model_name"] = tagger
            _TAGGER_CACHE["loaded_model"] = timm.create_model(f"hf_hub:{tagger}", pretrained=True)
            _TAGGER_CACHE["loaded_df"] = pd.read_csv(f"https://huggingface.co/{tagger}/resolve/main/selected_tags.csv")
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32 if dtype == "fp32" else torch.bfloat16
        _TAGGER_CACHE["loaded_model"] = _TAGGER_CACHE["loaded_model"].to("cuda", dtype=torch_dtype).eval()

        return io.NodeOutput(_TAGGER_CACHE["loaded_model"], _TAGGER_CACHE["loaded_df"])

class PredictTag(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"PredictTag{NODE_SURFIX}",
            display_name=f"Predict Tag {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                WDTagger.Input("tagger"),
                WDTaggerLabels.Input("labels"),
                io.Image.Input("image"),
                io.Boolean.Input("rating", default=False),
                io.Float.Input("character_thereshold", default=0.85, min=0.0, max=1.001, step=0.001),
                io.Float.Input("general_thereshold", default=0.35, min=0.0, max=1.001, step=0.001),
            ],
            outputs=[
                BatchString.Output(),
                io.String.Output(),
                WDTaggerFeatures.Output(),
            ],
        )

    @classmethod
    @torch.inference_mode(False)
    def execute(cls, tagger, labels, image, rating, character_thereshold, general_thereshold) -> io.NodeOutput:
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
            "prob": probs
        }

        return io.NodeOutput(prompts, string, features)

class GradCam(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"GradCam{NODE_SURFIX}",
            display_name=f"Grad Cam {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                WDTagger.Input("tagger"),
                WDTaggerFeatures.Input("features"),
                io.String.Input("target_tag", default="", multiline=True),
                io.Float.Input("heat_map_alpha", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("intepolate", options=["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"], default="bilinear"),
                io.Boolean.Input("negative"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    @torch.inference_mode(False)
    def execute(cls, tagger, features, target_tag, heat_map_alpha, intepolate, negative) -> io.NodeOutput:

        image = features["image"]

        size = (image.shape[1], image.shape[2])
        target_ids = [features["tag_to_id"][tag.strip().replace(" ", "_")] for tag in target_tag.strip().strip(",").split(",")]

        features = features["feature"].detach().clone().requires_grad_(True)

        gradients = []
        if features.shape[1] == 1025: # eva02-large
            feature_size = 32
            channel_dim = 2
            hw_dim = 1
            features = features[:,1:]
        elif features.shape[1] == 1024 and len(features.shape) == 3: # vit-large
            feature_size = 32
            channel_dim = 2
            hw_dim = 1
        elif features.shape[1] == 1024: # convnext
            feature_size = 14
            channel_dim = 1
            hw_dim = (2, 3)
        elif features.shape[2] == 768: # vit
            feature_size = 28
            channel_dim = 2
            hw_dim = 1
        elif features.shape[3] == 1024: # swin
            feature_size = 14
            channel_dim = 3
            hw_dim = (1, 2)

        for i in range(len(features)):
            feature = features[i].unsqueeze(0)
            outputs = tagger.forward_head(feature).sigmoid()

            output = outputs[0, torch.tensor(target_ids)].sum(dim=-1)

            gradients.append(torch.autograd.grad(output, feature, retain_graph=True)[0])
            tagger.zero_grad()
            features.grad = None
        gradients = torch.cat(gradients)

        weight = torch.mean(gradients, dim=hw_dim, keepdim=True)
        if negative:
            weight = -weight
        heat_map = torch.sum(weight * features, dim=channel_dim).relu().reshape(-1, 1, feature_size, feature_size)
        heat_map = heat_map / heat_map.max(dim=2, keepdim=True).values.max(dim=3, keepdim=True).values

        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = heat_map.reshape(-1, 1).detach().float().cpu().numpy()

        c_map = plt.get_cmap("jet")
        heat_map = c_map(heat_map).reshape(-1, feature_size, feature_size, 4)[:,:,:,:3]
        heat_map = torch.from_numpy(heat_map)
        heat_map = heat_map.permute(0, 3, 1, 2)

        heat_map = torch.nn.functional.interpolate(heat_map, size=size, mode=intepolate)
        heat_map = heat_map.permute(0, 2, 3, 1)

        return io.NodeOutput(image * (1 - heat_map_alpha) + heat_map * heat_map_alpha)

class GradCamAuto(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"GradCamAuto{NODE_SURFIX}",
            display_name=f"Grad Cam Auto {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                WDTagger.Input("tagger"),
                WDTaggerFeatures.Input("features"),
                io.Float.Input("threshold", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Float.Input("heat_map_alpha", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("intepolate", options=["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"], default="bilinear"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    @torch.inference_mode(False)
    def execute(cls, tagger, features, threshold, heat_map_alpha, intepolate) -> io.NodeOutput:

        image = features["image"].detach().clone()
        if image.shape[0] > 1:
            raise ValueError("Batch size must be 1")

        size = (image.shape[1], image.shape[2])
        id_to_tag = {v:k for k,v in features["tag_to_id"].items()}
        features = features["feature"].detach().clone().requires_grad_(True)

        gradients = []
        if features.shape[1] == 1025: # eva02-large
            feature_size = 32
            channel_dim = 2
            hw_dim = 1
            features = features[:,1:]
        elif features.shape[1] == 1024 and len(features.shape) == 3: # vit-large
            feature_size = 32
            channel_dim = 2
            hw_dim = 1
        elif features.shape[1] == 1024: # convnext
            feature_size = 14
            channel_dim = 1
            hw_dim = (2, 3)
        elif features.shape[2] == 768: # vit
            feature_size = 28
            channel_dim = 2
            hw_dim = 1
        elif features.shape[3] == 1024: # swin
            feature_size = 14
            channel_dim = 3
            hw_dim = (1, 2)


        outputs = tagger.forward_head(features).sigmoid()
        target_ids = torch.where(outputs > threshold)[1].detach()
        outputs_filtered = outputs[0, torch.tensor(target_ids)]

        for output in outputs_filtered:
            gradients.append(torch.autograd.grad(output, features, retain_graph=True)[0])
        tagger.zero_grad()
        features.grad = None

        gradients = torch.cat(gradients)

        weight = torch.mean(gradients, dim=hw_dim, keepdim=True)
        heat_map = torch.sum(weight * features, dim=channel_dim).relu().reshape(-1, 1, feature_size, feature_size)
        heat_map = heat_map / heat_map.max(dim=2, keepdim=True).values.max(dim=3, keepdim=True).values

        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = heat_map.reshape(-1, 1).detach().float().cpu().numpy()

        c_map = plt.get_cmap("jet")
        heat_map = c_map(heat_map).reshape(-1, feature_size, feature_size, 4)[:,:,:,:3]
        heat_map = torch.from_numpy(heat_map)
        heat_map = heat_map.permute(0, 3, 1, 2)

        heat_map = torch.nn.functional.interpolate(heat_map, size=size, mode=intepolate)
        heat_map = heat_map.permute(0, 2, 3, 1)

        output_image = image * (1 - heat_map_alpha) + heat_map * heat_map_alpha
        images = [np.ascontiguousarray((image * 255).numpy().astype(np.uint8)) for image in output_image]

        for image, target_id in zip(images, target_ids):
            target_tag = id_to_tag[target_id.item()]
            score = outputs[0, target_id]
            cv2.putText(image, f"{target_tag}:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"{score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # sort by score
        image_score = [(image, output.item()) for image, output in zip(images, outputs_filtered)]
        image_score.sort(key=lambda x: x[1], reverse=True)
        images = [image for image, _ in image_score]

        output_image = torch.from_numpy(np.array(images))
        output_image = output_image.float() / 255
        return io.NodeOutput(output_image)

class GradPair(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"GradPair{NODE_SURFIX}",
            display_name=f"Grad Pair {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                WDTagger.Input("tagger"),
                WDTaggerFeatures.Input("features"),
                io.Float.Input("heat_map_alpha", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("intepolate", options=["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"], default="bilinear"),
                io.Boolean.Input("negative"),
            ],
            outputs=[
                io.Image.Output(),
                io.String.Output(),
            ],
        )

    @classmethod
    @torch.inference_mode(False)
    def execute(cls, tagger, features, heat_map_alpha, intepolate, negative) -> io.NodeOutput:

        prob_diff = (features["prob"][0] - features["prob"][1])
        prob_diff_data = pd.DataFrame({"label": features["tag_to_id"].keys(), "prob_diff": prob_diff})
        prob_diff_data = prob_diff_data.sort_values(by="prob_diff", ascending=False)

        top_20 = prob_diff_data.head(20)
        bottom_20 = prob_diff_data.tail(20).sort_values(by="prob_diff")

        output_string = f"Top 20 difference:\n{top_20.to_string(index=False)}\n ... \n:\n{bottom_20.to_string(index=False)}"

        image = features["image"]
        if image.shape[0] != 2:
            raise ValueError("Batch size must be 2")

        size = (image.shape[1], image.shape[2])

        features = features["feature"].detach().clone().requires_grad_(True)

        gradients = []
        if features.shape[1] == 1025: # eva02-large
            feature_size = 32
            channel_dim = 2
            hw_dim = 1
            features = features[:,1:]
        elif features.shape[1] == 1024 and len(features.shape) == 3: # vit-large
            feature_size = 32
            channel_dim = 2
            hw_dim = 1
        elif features.shape[1] == 1024: # convnext
            feature_size = 14
            channel_dim = 1
            hw_dim = (2, 3)
        elif features.shape[2] == 768: # vit
            feature_size = 28
            channel_dim = 2
            hw_dim = 1
        elif features.shape[3] == 1024: # swin
            feature_size = 14
            channel_dim = 3
            hw_dim = (1, 2)

        for i in range(2):
            feature = features[i].unsqueeze(0)
            output_1 = tagger.forward_head(feature, pre_logits=True)
            with torch.no_grad():
                output_2 = tagger.forward_head(features[1-i].unsqueeze(0), pre_logits=True)

            sim = torch.nn.functional.cosine_similarity(output_1, output_2)
            gradients.append(torch.autograd.grad(sim, feature, retain_graph=True)[0])
            tagger.zero_grad()
            features.grad = None

        gradients = torch.cat(gradients)

        weight = torch.mean(gradients, dim=hw_dim, keepdim=True)
        if negative:
            weight = -weight
        heat_map = torch.sum(weight * features, dim=channel_dim).relu().reshape(-1, 1, feature_size, feature_size)
        heat_map = heat_map / heat_map.max(dim=2, keepdim=True).values.max(dim=3, keepdim=True).values

        heat_map = heat_map.permute(0, 2, 3, 1)
        heat_map = heat_map.reshape(-1, 1).detach().float().cpu().numpy()

        c_map = plt.get_cmap("jet")
        heat_map = c_map(heat_map).reshape(-1, feature_size, feature_size, 4)[:,:,:,:3]
        heat_map = torch.from_numpy(heat_map)
        heat_map = heat_map.permute(0, 3, 1, 2)

        heat_map = torch.nn.functional.interpolate(heat_map, size=size, mode=intepolate)
        heat_map = heat_map.permute(0, 2, 3, 1)

        return io.NodeOutput(image * (1 - heat_map_alpha) + heat_map * heat_map_alpha, output_string)

class WDTaggerSimilarity(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"WDTaggerSimilarity{NODE_SURFIX}",
            display_name=f"WD Tagger Similarity {SYMBOL}",
            category=CATEGORY_NAME,
            inputs=[
                WDTagger.Input("tagger"),
                WDTaggerLabels.Input("labels"),
                io.String.Input("tag", multiline=True),
                io.Combo.Input("category", options=["all", "general", "character"]),
                io.Boolean.Input("ascending", default=False),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, tagger, labels, tag, category, ascending) -> io.NodeOutput:
        dtype = tagger.parameters().__next__().dtype
        tag_list = [t.strip().replace(" ", "_") for t in tag.strip().strip(",").split(",")]
        tag_ids = [labels[labels["name"] == t].index[0] for t in tag_list if t in labels["name"].values]
        if len(tag_ids) == 0:
            return io.NodeOutput(f"No valid tags found in input: {tag}")

        with torch.no_grad():
            tag_embeddings = tagger.get_classifier().weight[tag_ids].to("cpu", dtype=dtype)
            all_embeddings = tagger.get_classifier().weight.to("cpu", dtype=dtype)

            tag_embeddings = tag_embeddings / tag_embeddings.norm(dim=1, keepdim=True)
            all_embeddings = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)

            similarity = torch.matmul(all_embeddings, tag_embeddings.T).min(dim=1).values.cpu().numpy()

        labels["similarity"] = similarity
        if category == "general":
            labels = labels[labels["category"] == 0]
        elif category == "character":
            labels = labels[labels["category"] == 4]

        labels = labels.sort_values(by="similarity", ascending=ascending)
        output_string = f"Similarity result for tags: {', '.join(tag_list)}\n"
        output_string += labels[["name", "similarity"]].head(50).to_string(index=False)

        return io.NodeOutput(output_string)
