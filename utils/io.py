import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Union
from PIL import Image, ExifTags
import cv2


def load_image(path: Union[str, Path]) -> npt.NDArray:
    # PIL is preferred but some images are rotated and EXIF should be used to
    # align them. OpenCV can do this automatically
    # image = Image.open(path)

    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.asarray(image).astype(np.uint8)


def load_label(path: Union[str, Path], binary: bool = True) -> npt.NDArray:
    image = Image.open(path)
    if not binary:
        return image
    image = np.asarray(image)[:, :, 0] > 0
    return image.astype(np.uint8)
