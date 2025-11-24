from src.utils.reshape_transforms import *

import torch.nn as nn

def find_last_conv_layer(model):
    """
    Fallback: find the last Conv2d or Conv3d layer in any model (works for 2D/3D).
    """
    last_conv = None
    last_name = None
    try:
        for name, module in model.named_modules():
            # ✅ Support both 2D and 3D conv layers
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                last_conv = module
                last_name = name
            elif getattr(module, "original_name", None) in ("Conv2d", "Conv3d"):  # TorchScript case
                last_conv = module
                last_name = name
    except Exception:
        pass

    if last_conv is None:
        print("[WARN] No Conv2d/Conv3d layer found — Grad-CAM may not work.")
    else:
        print(f"[INFO] Using last conv layer: {last_name} ({last_conv.__class__.__name__})")
    return last_conv


def get_hidden_layer(model, modality):

    """
    Return layer(s), reshape function, and include_negative flag.
    If model type is not recognized, fallback to last Conv2d automatically.
    """
    layer, reshape, include_negative = None, None, False

    if model.__class__.__name__ == "ResNet":
        layer = [model.layer4[-1]]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "EfficientNet":
        layer = [model.features[-1]]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "VisionTransformer":
        layer = [model.blocks[-1].norm1]
        reshape = reshape_transform_2D if modality == "image" else reshape_transform_3D
        include_negative = False
    elif model.__class__.__name__ == "EfficientNet3D":
        layer = [model._blocks[-13]]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "VideoResNet":
        layer = [model.layer3]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "PointNet":
        layer = [model.transform.bn1]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "DGCNN":
        layer = [model.conv5]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "PCT":
        layer = [model.pt_last.sa4.after_norm]
        reshape = reshape_transform_2D
        include_negative = True
    else:
      
        # ✅ Fallback for unknown architectures, now supporting 3D
        fallback = find_last_conv_layer(model)
        if fallback is not None:
            layer = [fallback]
            print(f"[INFO] Fallback layer selected: {fallback}")
        else:
            print("[WARN] No Conv2d/Conv3d layer found. Returning None.")

    # ✅ Always return list for compatibility
    return  layer , reshape, include_negative    


def get_hidden_layer_eval(model):

    layer = None  # ✅ ensures it's always defined

    if model.__class__.__name__ == "ResNet":
        layer = ["layer4.1.conv2"]
    elif model.__class__.__name__ == "EfficientNet":
        layer = ["features.8.0"]
    elif model.__class__.__name__ == "VisionTransformer":
        layer = ["blocks.11.norm1"]
    elif model.__class__.__name__ == "EfficientNet3D":
        layer = ["_blocks.15._expand_conv"]
    elif model.__class__.__name__ == "VideoResNet":
        layer = ["layer4.0.conv1.1"]
    elif model.__class__.__name__ == "PointNet":
        layer = ["transform.bn1"]
    elif model.__class__.__name__ == "DGCNN":
        layer = ["linear1"]
    elif model.__class__.__name__ == "PCT":
        layer = ["linear1"]

    # --- fallback: find last Conv2d or Conv3d layer ---
    if layer is None:
        last_name = None
        try:
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                    last_name = name
            if last_name is not None:
                print(f"[INFO] Fallback eval layer name: {last_name}")
                layer = [last_name]
            else:
                print("[WARN] No Conv2d/Conv3d layer found for evaluation — returning None.")
                layer = [None]
        except Exception as e:
            print(f"[WARN] Fallback failed: {e}")
            layer = [None]

    return layer