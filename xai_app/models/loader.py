# xai_app/models/loader.py
from dataclasses import dataclass
from typing import Optional, List

import os
import shutil
import time
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from transformers import AutoModelForImageClassification, AutoImageProcessor

from monai.networks.nets import DenseNet121 as MonaiDenseNet121
from monai.networks.nets import EfficientNetBN, ViT as MonaiViT

from safetensors.torch import load_file as load_safetensors


# ----------------------------
# Device selection
# ----------------------------
if torch.backends.mps.is_available() and torch.backends.mps.is_built():  # Apple Silicon
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"[INFO][models.loader] Using device: {DEVICE}")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    kind: str
    target_layer: Optional[torch.nn.Module]
    categories: Optional[List[str]] = None


# ----------------------------
# Helper to find last conv layer
# ----------------------------
def find_last_conv(m: nn.Module) -> Optional[nn.Module]:
    last_conv = None
    for _, sub in m.named_modules():
        if isinstance(sub, (nn.Conv2d, nn.Conv3d)):
            last_conv = sub
    if last_conv is None:
        print("[WARN] No Conv2d/Conv3d layer found in model!")
    else:
        print(f"[INFO] Found last conv layer: {last_conv.__class__.__name__}")
    return last_conv


# ----------------------------
# Demo model: torchvision DenseNet121
# ----------------------------
def load_demo_model() -> LoadedModel:
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights).to(DEVICE).eval()
    target_layer = find_last_conv(model)
    categories = weights.meta.get("categories") if hasattr(weights, "meta") else None
    return LoadedModel(model=model, kind="demo_densenet121", target_layer=target_layer, categories=categories)


# ----------------------------
# TorchScript loader
# ----------------------------
def load_torchscript(path: str) -> LoadedModel:
    t0 = time.perf_counter()
    fast_path = "/dev/shm/upload_model.pt"
    try:
        shutil.copy2(path, fast_path)
        load_path = fast_path
    except Exception as e:
        print(f"[load_torchscript] copy failed ({e}), fallback to {path}")
        load_path = path

    model = torch.jit.load(load_path, map_location="cpu").eval()

    if DEVICE == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        print("[INFO] TorchScript model moved to CUDA")
    else:
        model = model.to(DEVICE)
        print(f"[INFO] TorchScript model kept on {DEVICE}")

    elapsed = time.perf_counter() - t0
    print(f"[load_torchscript] Finished load in {elapsed:.2f}s")

    target_layer = find_last_conv(model)
    return LoadedModel(model=model, kind="torchscript", target_layer=target_layer, categories=None)


# ----------------------------
# Torchvision models
# ----------------------------
AVAILABLE_TORCHVISION_MODELS = [
    "resnet18", "resnet50", "densenet121", "mobilenet_v2",
    "efficientnet_b0", "vit_b_16", "swin_t", "convnext_tiny",
]


def load_torchvision_model(model_name: str) -> LoadedModel:
    if not hasattr(models, model_name):
        raise ValueError(f"torchvision has no model '{model_name}'")

    constructor = getattr(models, model_name)
    try:
        weights_attr = getattr(models, f"{model_name}_Weights", None)
        weights = weights_attr.DEFAULT if weights_attr is not None else None
        model = constructor(weights=weights).to(DEVICE).eval()
        categories = weights.meta.get("categories") if weights and hasattr(weights, "meta") else None
    except Exception:
        model = constructor(pretrained=True).to(DEVICE).eval()
        categories = None

    target_layer = find_last_conv(model)
    return LoadedModel(model=model, kind=f"torchvision:{model_name}", target_layer=target_layer, categories=categories)


# ----------------------------
# MONAI models
# ----------------------------
MONAI_MODELS = ["densenet121", "efficientnet_b0", "vit"]


def load_monai_model(name: str) -> LoadedModel:
    if name == "densenet121":
        model = MonaiDenseNet121(spatial_dims=2, in_channels=3, out_channels=2)
    elif name == "efficientnet_b0":
        model = EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=3, num_classes=2)
    elif name == "vit":
        model = MonaiViT(in_channels=3, img_size=(224, 224), patch_size=16, num_classes=2)
    else:
        raise ValueError(f"Unknown MONAI model '{name}'")

    model = model.to(DEVICE).eval()
    target_layer = find_last_conv(model)
    return LoadedModel(model=model, kind=f"monai:{name}", target_layer=target_layer, categories=None)


# ----------------------------
# HuggingFace transformer models
# ----------------------------
class HFModelWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        out = self.hf_model(x)
        if hasattr(out, "logits"):
            return out.logits
        return out


def load_transformer_model(model_name: str) -> LoadedModel:
    hf_model = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE).eval()
    wrapped = HFModelWrapper(hf_model)
    _ = AutoImageProcessor.from_pretrained(model_name)  # kept for later use if needed

    print(f"[INFO] Grad-CAM disabled for pure transformer model {model_name}")
    categories = hf_model.config.id2label if hasattr(hf_model.config, "id2label") else None
    return LoadedModel(model=wrapped, kind=f"transformer:{model_name}", target_layer=None, categories=categories)


# ----------------------------
# Custom architecture loader
# ----------------------------
def load_custom_architecture(py_file: str, st_file: Optional[str] = None) -> LoadedModel:
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location("custom_model", py_file)
    module = importlib.util.module_from_spec(spec)
    _sys.modules["custom_model"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if not hasattr(module, "build_model"):
        raise RuntimeError("Python file must define build_model() returning nn.Module.")

    model = module.build_model().to(DEVICE).eval()

    if st_file is not None:
        state_dict = load_safetensors(st_file)
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded weights from {st_file}")

    target_layer = find_last_conv(model)
    return LoadedModel(model=model, kind="custom_architecture", target_layer=target_layer, categories=None)
