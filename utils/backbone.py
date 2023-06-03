import torch
from typing import Literal
import torch.nn.functional as F
from utils.config import RESIZE


def get_backbone(size: Literal["s", "b", "l"] = "s") -> "Backbone":
    backbone = Backbone(size=size)
    return backbone


class Backbone(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{size}14")

    def get_features(self, x):
        # X is a tensor encoded as C, H, W
        with torch.no_grad():
            patch_tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]
            n_tokens, dim = patch_tokens.shape[1], patch_tokens.shape[-1]
            h = x.shape[-2] // 14
            w = x.shape[-1] // 14
            patch_tokens = patch_tokens.reshape(-1, h, w, dim)
            patch_tokens = patch_tokens.permute(0, 3, 2, 1)  # channel first
        return patch_tokens
