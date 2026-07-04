import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ... import ROOT_NAME, SYMBOL, NODE_SURFIX
from comfy_api.v0_0_2 import io

WDTaggerFeatures = io.Custom("WD-TAGGER-FEATURES")

def heatmap_to_numpy(heatmap, cmap="jet"):
    norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))  # 正規化
    colormap = plt.get_cmap(cmap)
    heatmap_rgb = colormap(norm(heatmap))[:, :, :3]
    return heatmap_rgb

class MSEHeatmap(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"MSEHeatmap{NODE_SURFIX}",
            display_name=f"MSE Heatmap {SYMBOL}",
            category=ROOT_NAME + "for_test",
            inputs=[
                io.Latent.Input("latent1"),
                io.Latent.Input("latent2"),
                io.Image.Input("image"),
                io.Float.Input("alpha", default=0.3, min=0, max=1, step=0.01),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, latent1, latent2, image, alpha) -> io.NodeOutput:
        latent1 = latent1["samples"]
        latent2 = latent2["samples"]
        print(latent1.size(), latent2.size(), image.size())
        error = torch.norm(latent1 - latent2, dim=1, keepdim=False)
        heatmaps = [heatmap_to_numpy(error[i].cpu().numpy()) for i in range(error.size(0))]
        heatmaps = torch.from_numpy(np.array(heatmaps))
        h, w = image.size(1), image.size(2)
        print(heatmaps.size())
        heatmaps = heatmaps.permute(0, 3, 1, 2)
        heatmaps = torch.nn.functional.interpolate(heatmaps, size=(h, w), mode="bilinear")
        heatmaps = heatmaps.permute(0, 2, 3, 1)
        print(heatmaps.size())
        heatmaps = heatmaps * alpha + image * (1 - alpha)
        return io.NodeOutput(heatmaps)

class MSEHeatmapTagger(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"MSEHeatmapTagger{NODE_SURFIX}",
            display_name=f"MSE Heatmap Tagger {SYMBOL}",
            category=ROOT_NAME + "for_test",
            inputs=[
                WDTaggerFeatures.Input("features"),
                io.Image.Input("image"),
                io.Float.Input("alpha", default=0.3, min=0, max=1, step=0.01),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, features, image, alpha) -> io.NodeOutput:
        features = features["feature"].detach().clone().cpu()
        bsz = features.shape[0]
        if features.shape[1] == 1025: # eva02-large
            feature_size = 32
            channel_dim = 2
            hw_dim = 1
            features = features[:,1:]
            features = features.view(bsz, feature_size, feature_size, -1).permute(0, 3, 1, 2)
        elif features.shape[1] == 1024 and len(features.shape) == 3: # vit-large
            feature_size = 32
            channel_dim = 2
            hw_dim = 1
            features = features.view(bsz, feature_size, feature_size, -1).permute(0, 3, 1, 2)
        elif features.shape[1] == 1024: # convnext
            feature_size = 14
            channel_dim = 1
            hw_dim = (2, 3)
            features = features.view(bsz, -1, feature_size, feature_size)
        elif features.shape[2] == 768: # vit
            feature_size = 28
            channel_dim = 2
            hw_dim = 1
            features = features.view(bsz, feature_size, feature_size, -1).permute(0, 3, 1, 2)
        elif features.shape[3] == 1024: # swin
            feature_size = 14
            channel_dim = 3
            hw_dim = (1, 2)
            features = features.permute(0, 3, 1, 2)

        print(features.size(),  image.size())
        error = torch.norm(features[:1] - features[1:], dim=1, keepdim=False)
        heatmaps = [heatmap_to_numpy(error[i].cpu().numpy()) for i in range(error.size(0))]
        heatmaps = torch.from_numpy(np.array(heatmaps))
        h, w = image.size(1), image.size(2)
        print(heatmaps.size())
        heatmaps = heatmaps.permute(0, 3, 1, 2)
        heatmaps = torch.nn.functional.interpolate(heatmaps, size=(h, w), mode="bilinear")
        heatmaps = heatmaps.permute(0, 2, 3, 1)
        print(heatmaps.size())
        heatmaps = heatmaps * alpha + image[1:] * (1 - alpha)
        return io.NodeOutput(heatmaps)
