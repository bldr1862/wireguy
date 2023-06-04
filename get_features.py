from pathlib import Path
from utils import backbone as BKB
import pandas as pd
from utils.device import DEVICE
from utils.transforms import TRANSFORM
from utils.dataset import TorchDataset
from torch.utils.data import DataLoader
import torch

DATA_PATH = "data/ISG-Guywire_Ground_Seg/"
FEATURES_DIR = "data/features"

# get images and labels paths
images_files = [x for x in Path(DATA_PATH).glob("*.JPG")]
labels_files = [x for x in Path(DATA_PATH).glob("*.png")]

print(f"Number of images {len(images_files)}")
print(f"Number of labels {len(labels_files)}")

images_files = pd.DataFrame(images_files, columns=["image_path"])
labels_files = pd.DataFrame(labels_files, columns=["label_path"])

images_files["stem"] = images_files["image_path"].apply(lambda x: x.stem)
labels_files["stem"] = labels_files["label_path"].apply(lambda x: x.stem)

consolidation = images_files.merge(labels_files, how="inner", on="stem")
print(f"Dataset size {len(consolidation)}")

images_files = [x for x in Path(DATA_PATH).glob("*.JPG")]
backbone = BKB.get_backbone(size="s")

backbone.eval()
backbone.to(DEVICE)

dataset = TorchDataset(consolidation, transform=TRANSFORM, return_raw=True)
dataloader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=1)

Path(FEATURES_DIR).mkdir(exist_ok=True, parents=True)
for i, data in enumerate(dataloader):
    print(f"Running {i} out of {len(dataloader)}")
    image_transform, image, label, image_path = data
    with torch.no_grad():
        features = backbone.get_features(image_transform.to(DEVICE))
    out_file = Path(FEATURES_DIR).joinpath(f"{Path(image_path[0]).stem}.pt")
    torch.save(features, out_file)
