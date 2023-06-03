from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from .model import SegmentationModel
from .device import DEVICE


def train_one_epoch(
    model: SegmentationModel,
    train_dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: _Loss,
):
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    model.backbone.eval()
    model.head.train()
    model.to(DEVICE)
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs.to(DEVICE))

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels.to(DEVICE).long())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
    return running_loss / len(train_dataloader)
