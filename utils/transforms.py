from torchvision import transforms as tr
import cv2
import numpy.typing as npt
import numpy as np


def resize_label(label: npt.NDArray):
    resize = cv2.resize(label, dsize=RESIZE, interpolation=cv2.INTER_NEAREST)
    return np.asarray(resize)


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
RESIZE = (560, 560)  # DinoV2 Vit needs a multiple of 14

TRANSFORM = tr.Compose(
    [
        tr.ToTensor(),
        tr.Resize(RESIZE),
        tr.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
)


TARGET_TRANSFORM = resize_label
