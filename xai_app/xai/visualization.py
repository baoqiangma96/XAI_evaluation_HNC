# xai_app/xai/visualization.py
import io
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def overlay_heatmap(pil_img: Image.Image, heat: np.ndarray, alpha: float = 0.5) -> Image.Image:
    if heat.ndim == 4:
        heat = np.mean(heat, axis=(0, 1))
    if heat.ndim == 3 and heat.shape[0] in (1, 3):
        heat = np.mean(heat, axis=0)

    h, w = heat.shape
    img = pil_img.resize((w, h))

    heat = (heat - np.min(heat)) / (np.max(heat) - np.min(heat) + 1e-8)

    fig = plt.figure(figsize=(4, 4), dpi=200)
    plt.imshow(img)
    plt.imshow(heat, cmap="jet", alpha=alpha)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def overlay_heatmap_3d(
    volume,
    amap,
    view: Literal["axial", "coronal", "sagittal"] = "axial",
    slice_idx: int = 48,
    alpha: float = 0.5,
    channel_mode: str = "Mean (all)",
) -> Image.Image:
    volume = np.asarray(volume)
    amap = np.asarray(amap)

    if volume.ndim == 3:
        volume = volume[np.newaxis, ...]
    if amap.ndim == 3:
        amap = amap[np.newaxis, ...]

    C, D, H, W = volume.shape

    ch = 0
    if "PET" in channel_mode:
        ch = 1
    elif "GTV" in channel_mode:
        ch = 2

    base = np.mean(volume, axis=0) if "Mean" in channel_mode else volume[min(ch, C - 1)]

    if view == "axial":
        max_idx = D - 1
        s = int(np.clip(slice_idx, 0, max_idx))
        base2d = base[s]
        amap2d = amap[min(ch, amap.shape[0] - 1), s] if amap.ndim == 4 else amap[s]
    elif view == "coronal":
        max_idx = H - 1
        s = int(np.clip(slice_idx, 0, max_idx))
        base2d = base[:, s, :]
        amap2d = amap[min(ch, amap.shape[0] - 1), :, s, :] if amap.ndim == 4 else amap[:, s, :]
    elif view == "sagittal":
        max_idx = W - 1
        s = int(np.clip(slice_idx, 0, max_idx))
        base2d = base[:, :, s]
        amap2d = amap[min(ch, amap.shape[0] - 1), :, :, s] if amap.ndim == 4 else amap[:, :, s]
    else:
        raise ValueError(f"Unknown view: {view}")

    base2d = (base2d - np.min(base2d)) / (np.ptp(base2d) + 1e-8)
    amap2d = (amap2d - np.min(amap2d)) / (np.ptp(amap2d) + 1e-8)

    rgb = plt.cm.jet(amap2d)[..., :3]
    overlay = (1 - alpha) * np.stack([base2d] * 3, axis=-1) + alpha * rgb
    overlay = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay)


def add_label_to_image(pil_img: Image.Image, text: str) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), text, fill="white")
    return img
