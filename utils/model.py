from torch import nn
import torch
from utils import backbone as BKB
from utils import head as HEAD
from utils.config import RESIZE
import torch.nn.functional as F


class SegmentationModel(nn.Module):
    def __init__(self, backbone: BKB.Backbone, head: HEAD.ConvHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.get_features(x)
            features = F.interpolate(features, RESIZE, mode="bilinear")
        logits = self.head(features)
        return logits
