# xai_app/xai/helpers.py
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image, ImageDraw

from xai_app.xai.visualization import overlay_heatmap_3d, add_label_to_image


def make_error_image(msg: str) -> Image.Image:
    img = Image.new("RGB", (640, 200), "white")
    d = ImageDraw.Draw(img)
    d.text((10, 10), ("Error: " + msg)[:220], fill="red")
    return img


def _flatten_cache_for_viewer(cache_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Accepts:
      - flat: {method: {"input":..., "amap":...}}
      - nested: {patient_id: {method: {"input":..., "amap":...}}}
    Returns a flat dict keyed as:
      - method
      - or "method | patient_id"
    """
    if not isinstance(cache_dict, dict):
        return {}

    # flat?
    if all(isinstance(v, dict) and ("input" in v and "amap" in v)
           for v in cache_dict.values()):
        return cache_dict

    flat: Dict[str, Dict[str, Any]] = {}
    for pid, mdict in cache_dict.items():
        if not isinstance(mdict, dict):
            continue
        suffix = f" | {Path(str(pid)).stem}"
        for mname, entry in mdict.items():
            if isinstance(entry, dict) and ("input" in entry and "amap" in entry):
                flat[f"{mname}{suffix}"] = entry
    return flat


def update_3d_view(
    cache_dict: Dict[str, Any],
    method_names: List[str],
    view: str,
    slice_idx: int,
    alpha: float,
    channel_mode: str,
):
    """
    Multi-method 3D viewer.
    - Accepts flat or nested caches.
    - Synchronizes slice/view/channel/alpha.
    """
    flat_cache = _flatten_cache_for_viewer(cache_dict)
    if not flat_cache:
        print("[INFO] Empty or invalid cache; skip update.")
        return []

    if isinstance(method_names, str):
        method_names = [method_names]

    selected_keys: List[str] = []
    if method_names:
        lower_targets = [m.lower() for m in method_names]
        for k in flat_cache.keys():
            kl = k.lower()
            if any(kl.startswith(t) for t in lower_targets):
                selected_keys.append(k)
    if not selected_keys:
        selected_keys = list(flat_cache.keys())

    imgs_out = []
    slice_idx = int(slice_idx)

    for k in selected_keys:
        entry = flat_cache.get(k)
        if not entry:
            continue
        vol = entry.get("input")
        amap = entry.get("amap")
        if vol is None or amap is None:
            continue

        try:
            img = overlay_heatmap_3d(
                vol,
                amap,
                view=view,
                slice_idx=slice_idx,
                alpha=alpha,
                channel_mode=channel_mode,
            )
            img = add_label_to_image(img, f"{k} | {view}:{slice_idx}")
            imgs_out.append(img)
        except Exception as e:
            print(f"[WARN] update_3d_view failed for {k}: {e}")
            continue

    return imgs_out


def save_xai_maps(cache_dict: Dict[str, Any]):
    """
    Flatten nested cache (patient -> method -> {input, amap})
    and save all maps to a single .npz file.
    """
    import numpy as np

    if not cache_dict:
        raise ValueError("No XAI maps found.")

    all_inputs, all_amaps = {}, {}

    for pid, mdict in cache_dict.items():
        if not isinstance(mdict, dict):
            continue
        for mname, entry in mdict.items():
            if not isinstance(entry, dict):
                continue
            if "input" in entry and "amap" in entry:
                key_base = f"{mname}__{Path(str(pid)).stem}"
                all_inputs[f"{key_base}_input"] = entry["input"]
                all_amaps[f"{key_base}_amap"] = entry["amap"]

    if not all_inputs:
        raise ValueError("Cache is empty or in unexpected format.")

    out_path = "xai_maps_all.npz"
    np.savez_compressed(out_path, **all_inputs, **all_amaps)
    print(f"[INFO] Saved {len(all_inputs)} XAI maps to {out_path}")
    return out_path
