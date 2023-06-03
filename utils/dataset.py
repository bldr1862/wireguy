from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import utils.io as IO
import utils.config as C
import torch
from pathlib import Path


class TorchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: transforms = None,
        return_raw: bool = False,
    ):
        self.dataset = df
        self.transform = transform
        self.return_raw = return_raw

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image_path = self.dataset["image_path"].iloc[idx]
        label_path = self.dataset["label_path"].iloc[idx]

        image = IO.load_image(image_path, resize=C.RESIZE)
        label = IO.load_label(label_path, resize=C.RESIZE)

        if self.transform:
            image_transform = self.transform(image)
        if self.return_raw:
            return image_transform, image, label, str(image_path)
        return image_transform, label


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        return_stem: bool = False,
    ):
        self.dataset = df
        self.return_stem = return_stem

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # image_path = self.dataset["image_path"].iloc[idx]
        label_path = self.dataset["label_path"].iloc[idx]
        feature_path = self.dataset["feature_path"].iloc[idx]

        # image = IO.load_image(image_path, resize=C.RESIZE)
        label = IO.load_label(label_path, resize=C.RESIZE)
        feature = torch.load(feature_path)[0]  # remove batch size from feature tensor

        if self.return_stem:
            return feature, label, str(Path(feature_path).stem)
        return feature, label
