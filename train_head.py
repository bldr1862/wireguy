from pathlib import Path
from utils.device import DEVICE
from utils import head as HEAD
from PIL import Image
import torch
import pandas as pd
from utils.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from utils.trainer import train_head_one_epoch
from utils.eval import eval_head
import time
import numpy as np
import utils.config as C
import torch.nn.functional as F

ID = int(time.time() * 100)
DATA_PATH = "data/ISG-Guywire_Ground_Seg/"
FEATURES_PATH = "data/features"

# get images and labels paths
images_files = [x for x in Path(DATA_PATH).glob("*.JPG")]
labels_files = [x for x in Path(DATA_PATH).glob("*.png")]
features_files = [x for x in Path(FEATURES_PATH).glob("*.pt")]

print(f"Number of images {len(images_files)}")
print(f"Number of labels {len(labels_files)}")
print(f"Number of features {len(features_files)}")

images_files = pd.DataFrame(images_files, columns=["image_path"])
labels_files = pd.DataFrame(labels_files, columns=["label_path"])
features_files = pd.DataFrame(features_files, columns=["feature_path"])

images_files["stem"] = images_files["image_path"].apply(lambda x: x.stem)
labels_files["stem"] = labels_files["label_path"].apply(lambda x: x.stem)
features_files["stem"] = features_files["feature_path"].apply(lambda x: x.stem)

consolidation = images_files.merge(labels_files, how="inner", on="stem")
consolidation = consolidation.merge(features_files, how="inner", on="stem")
print(f"Dataset size {len(consolidation)}")

# split
dataset = consolidation.copy().sample(frac=1, random_state=324)
train_set = dataset.iloc[: int(len(dataset) * C.TRAIN_PERCENTAGE), :]
val_set = dataset.iloc[int(len(dataset) * C.TRAIN_PERCENTAGE) :, :]

print(f"training set: {len(train_set)}")
print(f"val set: {len(val_set)}")

# dataloaders
train_ds = EmbeddingDataset(train_set)
val_ds = EmbeddingDataset(val_set)

# pin memory false because tensors are already on gpu
train_dataloader = DataLoader(
    train_ds, shuffle=True, pin_memory=False, batch_size=C.BATCH_SIZE
)
val_dataloader = DataLoader(
    val_ds, shuffle=False, pin_memory=False, batch_size=C.BATCH_SIZE
)

# model
head = HEAD.get_head(n_classes=2, input_dims=384, layer_sizes=(64, 64))
head.to(DEVICE)

total_params = sum(p.numel() for p in head.parameters())
print(f"Total parameters {total_params}")

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(C.CE_WEIGHTS)).to(DEVICE)
optimizer = torch.optim.AdamW(head.parameters(), lr=0.0001)


for i in range(C.EPOCHS):
    print(f"Running epoch {i}")
    loss = train_head_one_epoch(head, train_dataloader, optimizer, loss_fn)
    f1_score = eval_head(head, val_dataloader)
    print(f"Train Loss: {loss} Val f1 score {f1_score}")


# predict
train_ds = EmbeddingDataset(train_set, return_stem=True)
val_ds = EmbeddingDataset(val_set, return_stem=True)
train_dataloader = DataLoader(
    train_ds,
    shuffle=False,
    pin_memory=False,
    batch_size=1,
)
val_dataloader = DataLoader(val_ds, shuffle=False, pin_memory=False, batch_size=1)

head.eval()
train_pred_dir = f"results/{ID}/train"
Path(train_pred_dir).mkdir(exist_ok=True, parents=True)
for i, data in enumerate(train_dataloader):
    inputs, label, stem = data
    with torch.no_grad():
        inputs = inputs.to(DEVICE)
        inputs = F.interpolate(inputs, C.RESIZE, mode="bilinear")
        preds = head(inputs)
        preds = preds.permute(0, 2, 3, 1)  # probs at the end
        preds = F.softmax(preds, dim=-1)
        pred_mask = torch.argmax(preds, dim=-1)
        pred_mask = pred_mask.detach().cpu().numpy()
        pred_mask = pred_mask * 255  # for easy viz
    dst_path = Path(train_pred_dir).joinpath(f"{stem[0]}.png")
    Image.fromarray(pred_mask[0].astype(np.uint8)).save(dst_path)

val_pred_dir = f"results/{ID}/val"
Path(val_pred_dir).mkdir(exist_ok=True, parents=True)
for i, data in enumerate(val_dataloader):
    inputs, label, stem = data
    with torch.no_grad():
        inputs = inputs.to(DEVICE)
        inputs = F.interpolate(inputs, C.RESIZE, mode="bilinear")
        preds = head(inputs)
        preds = preds.permute(0, 2, 3, 1)  # probs at the end
        preds = F.softmax(preds, dim=-1)
        pred_mask = torch.argmax(preds, dim=-1)
        pred_mask = pred_mask.detach().cpu().numpy()
        pred_mask = pred_mask * 255  # for easy viz

    dst_path = Path(val_pred_dir).joinpath(f"{stem[0]}.png")
    Image.fromarray(pred_mask[0].astype(np.uint8)).save(dst_path)

print(f"Finish training {ID}")
