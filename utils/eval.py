from utils.model import SegmentationModel
from utils.head import ConvHead
from torch.utils.data import DataLoader
from utils.device import DEVICE
import torch
from torcheval.metrics.classification import BinaryF1Score
import torch.nn.functional as F
from utils.config import RESIZE


def eval(model: SegmentationModel, dataloader: DataLoader):
    model.eval()
    metric = BinaryF1Score()
    for i, data in enumerate(dataloader):
        inputs, labels = data
        with torch.no_grad():
            preds = model(inputs.to(DEVICE))
            preds = preds.permute(0, 2, 3, 1)  # probs at the end
            preds = F.softmax(preds, dim=-1)
            preds = preds.detach().cpu()
            pred_mask = torch.argmax(preds, dim=-1)
            metric.update(pred_mask.flatten(), labels.flatten())
    return metric.compute()


def eval_head(model: ConvHead, dataloader: DataLoader):
    model.eval()
    metric = BinaryF1Score().to(DEVICE)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        with torch.no_grad():
            inputs = inputs.to(DEVICE)
            inputs = F.interpolate(inputs, RESIZE, mode="bilinear")
            preds = model(inputs)
            preds = preds.permute(0, 2, 3, 1)  # probs at the end
            preds = F.softmax(preds, dim=-1)
            pred_mask = torch.argmax(preds, dim=-1)
            pred_mask = pred_mask.detach().cpu()
            metric.update(pred_mask.flatten(), labels.flatten())
    return metric.compute()
