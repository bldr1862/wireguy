from pathlib import Path
from utils import io as IO
from utils.device import DEVICE
from utils import backbone as BKB
from utils import head as HEAD
from PIL import Image
from utils.transforms import TRANSFORM, TARGET_TRANSFORM
import torch
import pandas as pd
import cv2
from tqdm import tqdm
from utils.dataset import TorchDataset
from torch.utils.data import DataLoader
from utils.model import SegmentationModel
from utils.trainer import train_one_epoch
from utils.eval import eval
import time
import numpy as np

ID = int(time.time() * 100)
LABELS_PATH = "data/ISG-Guywire_Ground_Seg/"
IMAGES_PATH = ""

# get images and labels paths
images_files = [x for x in Path(LABELS_PATH).glob("*.JPG")]
labels_files = [x for x in Path(LABELS_PATH).glob("*.png")]

print(f"Number of images {len(images_files)}")
print(f"Number of labels {len(labels_files)}")

images_files = pd.DataFrame(images_files, columns=["image_path"])
labels_files = pd.DataFrame(labels_files, columns=["label_path"])

images_files["stem"] = images_files["image_path"].apply(lambda x: x.stem)
labels_files["stem"] = labels_files["label_path"].apply(lambda x: x.stem)

consolidation = images_files.merge(labels_files, how="inner", on="stem")
print(f"Dataset size {len(consolidation)}")

# split
TRAIN_PERCENTAGE = 0.8
dataset = consolidation.copy().sample(frac=1, random_state=324)
train_set = dataset.iloc[: int(len(dataset) * TRAIN_PERCENTAGE), :]
val_set = dataset.iloc[int(len(dataset) * TRAIN_PERCENTAGE) :, :]

print(f"training set: {len(train_set)}")
print(f"val set: {len(val_set)}")

# dataloaders
train_ds = TorchDataset(
    train_set, transform=TRANSFORM, target_transform=TARGET_TRANSFORM
)
val_ds = TorchDataset(val_set, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)

BATCH_SIZE = 2
train_dataloader = DataLoader(
    train_ds, shuffle=True, pin_memory=True, batch_size=BATCH_SIZE
)
val_dataloader = DataLoader(
    val_ds, shuffle=False, pin_memory=True, batch_size=BATCH_SIZE
)

# initialize model
# 2 classes, background and guy wire
# backbone size S -> 384
backbone = BKB.get_backbone(size="s")
head = HEAD.get_head(n_classes=2, input_dims=384, layer_sizes=(64, 64))

backbone.eval()
backbone.to(DEVICE)

head.to(DEVICE)

seg_model = SegmentationModel(backbone=backbone, head=head)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(seg_model.parameters(), lr=0.001)


EPOCHS = 10
for i in range(EPOCHS):
    print(f"Running epoch {i}")
    loss = train_one_epoch(seg_model, train_dataloader, optimizer, loss_fn)
    f1_score = eval(seg_model, val_dataloader)
    print(f"Val f1 score {f1_score}")

# predict
BATCH_SIZE = 1
train_ds = TorchDataset(
    train_set, transform=TRANSFORM, target_transform=TARGET_TRANSFORM, return_raw=True
)
val_ds = TorchDataset(
    val_set, transform=TRANSFORM, target_transform=TARGET_TRANSFORM, return_raw=True
)
train_dataloader = DataLoader(
    train_ds,
    shuffle=True,
    pin_memory=True,
    batch_size=BATCH_SIZE,
)
val_dataloader = DataLoader(
    val_ds, shuffle=False, pin_memory=True, batch_size=BATCH_SIZE
)

seg_model.eval()
train_pred_dir = f"results/train/{ID}"
Path(train_pred_dir).mkdir(exist_ok=True, parents=True)
for i, data in enumerate(train_dataloader):
    image_transform, label_transform, image, label, image_path = data
    with torch.no_grad():
        preds = seg_model(image_transform.to(DEVICE))
        preds = preds.detach().cpu()
        preds = preds.permute(0, 2, 3, 1)  # probs at the end
        pred_mask = torch.argmax(preds, dim=-1).numpy()
        pred_mask = pred_mask * 255  # for easy viz
    dst_path = Path(train_pred_dir).joinpath(f"{Path(image_path[0]).stem}.png")
    Image.fromarray(pred_mask[0].astype(np.uint8)).save(dst_path)

val_pred_dir = f"results/val/{ID}"
Path(val_pred_dir).mkdir(exist_ok=True, parents=True)
for i, data in enumerate(val_dataloader):
    image_transform, label_transform, image, label, image_path = data
    with torch.no_grad():
        preds = seg_model(image_transform.to(DEVICE))
        preds = preds.detach().cpu()
        preds = preds.permute(0, 2, 3, 1)  # probs at the end
        pred_mask = torch.argmax(preds, dim=-1).numpy()
        pred_mask = pred_mask * 255  # for easy viz

    dst_path = Path(val_pred_dir).joinpath(f"{Path(image_path[0]).stem}.png")
    Image.fromarray(pred_mask[0].astype(np.uint8)).save(dst_path)

print(f"Finish training {ID}")
