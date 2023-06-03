from torch import nn
import torch
from . import backbone as BKB
from . import head as HEAD


class SegmentationModel(nn.Module):
    def __init__(self, backbone: BKB.Backbone, head: HEAD.ConvHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.get_features(x)
        logits = self.head(features)
        return logits
