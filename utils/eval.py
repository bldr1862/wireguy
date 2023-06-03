from .model import SegmentationModel
from torch.utils.data import DataLoader
from .device import DEVICE
import torch
from torcheval.metrics.classification import BinaryF1Score


def eval(model: SegmentationModel, dataloader: DataLoader):
    model.eval()
    metric = BinaryF1Score()
    for i, data in enumerate(dataloader):
        inputs, labels = data
        with torch.no_grad():
            preds = model(inputs.to(DEVICE))
            preds = preds.detach().cpu()
            preds = preds.permute(0, 2, 3, 1)  # probs at the end
            pred_mask = torch.argmax(preds, dim=-1)
            metric.update(pred_mask.flatten(), labels.flatten())
    return metric.compute()
