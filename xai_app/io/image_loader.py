# xai_app/io/image_loader.py
from typing import Tuple

from PIL import Image
import torch
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _as_3ch_pil(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def load_image_2d(path: str) -> Image.Image:
    return _as_3ch_pil(Image.open(path).convert("RGB"))


def preprocess_both(pil_img: Image.Image, device: str = "cpu") -> Tuple[torch.Tensor, Image.Image]:
    """
    Return:
      x:  tensor of shape (1,3,224,224) on device
      img_cropped: preprocessed PIL image used for visualization
    """
    img_resized = TF.resize(pil_img, 256)
    img_cropped = TF.center_crop(img_resized, 224)
    x = TF.to_tensor(img_cropped)
    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x.unsqueeze(0).to(device), img_cropped
