from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import utils.io as IO


class TorchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: transforms = None,
        target_transform: transforms = None,
        return_raw: bool = False,
    ):
        self.dataset = df
        self.transform = transform
        self.target_transform = target_transform
        self.return_raw = return_raw

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image_path = self.dataset["image_path"].iloc[idx]
        label_path = self.dataset["label_path"].iloc[idx]

        image = IO.load_image(image_path)
        label = IO.load_label(label_path)

        if self.transform:
            image_transform = self.transform(image)
        if self.target_transform:
            label_transform = self.target_transform(label)
        if self.return_raw:
            return image_transform, label_transform, image, label, str(image_path)
        return image_transform, label_transform
