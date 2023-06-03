from torchvision import transforms as tr
import cv2
import numpy.typing as npt
import numpy as np

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

TRANSFORM = tr.Compose(
    [
        tr.ToTensor(),
        tr.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
)
