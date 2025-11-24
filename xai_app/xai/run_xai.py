# xai_app/xai/run_xai.py
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr

from xai_app.io.image_loader import load_image_2d, preprocess_both
from xai_app.io.volume_loader import load_nifti_slice, load_npy_volume
from xai_app.models.loader import LoadedModel
from xai_app.utils.config_loader import load_latec_config
from xai_app.utils.paths import ensure_latec_on_syspath
from xai_app.xai.visualization import overlay_heatmap_3d, add_label_to_image
from xai_app.xai.helpers import make_error_image


# --- LATEC modules (we assume third_party/latec/src is on sys.path) ---
ensure_latec_on_syspath()
from src.modules.xai_methods import XAIMethodsModule  # type: ignore


def _predict_topk(model: torch.nn.Module, x: torch.Tensor, categories=None, k: int = 5):
    with torch.no_grad():
        logits = model(x)
        if hasattr(logits, "logits"):
            logits = logits.logits
        if logits.ndim == 1:
            logits = logits[None]
        probs = F.softmax(logits, dim=-1)[0]
        topk = torch.topk(probs, min(k, probs.numel()))
        items = []
        for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            name = categories[idx] if categories and idx < len(categories) else f"class_{idx}"
            items.append({"index": int(idx), "label": name, "prob": float(p)})
    return items


def _prepare_input(
    data_file: str,
    is_nifti: bool,
    slice_idx: int,
    device: str,
) -> Tuple[torch.Tensor, Any, str]:
    """
    Returns:
      x: tensor to feed into model
      vis_obj: PIL image or None (for visualization)
      modality: "image" or "volume"
    """
    ext = str(data_file).lower()
    if ext.endswith(".npz") or ext.endswith(".npy"):
        x = load_npy_volume(data_file, device=device)
        return x, None, "volume"
    elif ext.endswith(".nii") or ext.endswith(".nii.gz") or is_nifti:
        pil_raw = load_nifti_slice(data_file, slice_idx)
        x, pil_vis = preprocess_both(pil_raw, device=device)
        return x, pil_vis, "volume"
    else:
        pil_raw = load_image_2d(data_file)
        x, pil_vis = preprocess_both(pil_raw, device=device)
        return x, pil_vis, "image"


def run_xai_single(
    model_state: Dict[str, Any],
    data_file: str,
    is_nifti: bool,
    slice_idx: int,
    methods: List[str],
    target_class: int,
    heat_alpha: float,
) -> Tuple[List[Any], Dict[str, Any], str]:
    """
    Run all selected XAI methods on a single input.
    Returns:
      - list of initial overlay images
      - cache dict
      - JSON string of predictions/top class info
    """
    if model_state is None or "_obj" not in model_state:
        raise gr.Error("Load or create a model first.")
    if data_file is None:
        raise gr.Error("Please upload an image, NIfTI, or preprocessed 3D npz/npy file.")

    lm: LoadedModel = model_state["_obj"]
    model = lm.model
    device = str(next(model.parameters()).device)

    x, _, modality = _prepare_input(data_file, is_nifti, slice_idx, device=device)
    model.eval()

    # Predict top-k
    preds = _predict_topk(model, x, categories=lm.categories, k=5)
    used_target = int(target_class) if (target_class is not None and target_class >= 0) else int(preds[0]["index"])

    # Load LATEC config (we force volume for 3D-like data)
    cfg = load_latec_config(modality="volume" if modality == "volume" else "image", config_name="explain")

    # Initialize XAI methods from LATEC
    xai_module = XAIMethodsModule(cfg, model, x)
    attr_dict: Dict[str, Any] = {}

    for method, hparams in zip(xai_module.xai_methods, xai_module.xai_hparams):
        mname = method.__class__.__name__.lower()
        if methods and mname not in [m.lower() for m in methods]:
            continue
        
        try:
            # Ensure tensors on same device
            x_device = x.to(device)
            model = model.to(device)
            if hasattr(method, "model"):
                method.model = method.model.to(device)
            if isinstance(hparams, dict):
                for k, v in hparams.items():
                    if torch.is_tensor(v):
                        hparams[k] = v.to(device)

            if mname in ["deepliftshap", "gradientshap"]:
                base = x.detach().clone()
                n_baselines = 10
                noise_std = 0.05 * (base.max() - base.min() + 1e-8)
                baselines = base.repeat(n_baselines, 1, 1, 1, 1)
                baselines += torch.randn_like(baselines) * noise_std
                hparams["baselines"] = baselines.to(x.device)

            if method.__class__.__name__.lower() in ["lime", "kernelshap"] :
                print("[INFO][LIME] Injecting explicit 3D feature_mask + baselines for 96Â³ input")

                def generate_feature_mask_3d_divided(x, n_parts=8):
                    """
                    Generate a 3D feature mask that divides each spatial dimension into `n_parts` equal chunks.
                    This automatically computes patch_size per dimension.
                    Example: for 96Â³ input and n_parts=8 â†’ patch_size = 12, total 512 patches.
                    """
                    _, c, d, h, w = x.shape

                    # Compute adaptive patch size per dimension
                    pd, ph, pw = d // n_parts, h // n_parts, w // n_parts

                    mask = torch.zeros((1, c, d, h, w), dtype=torch.long)
                    idx = 0
                    for z in range(0, d, pd):
                        for y in range(0, h, ph):
                            for x_ in range(0, w, pw):
                                mask[:, :, z:z+pd, y:y+ph, x_:x_+pw] = idx
                                idx += 1
                    return mask

                feature_mask = generate_feature_mask_3d_divided(x, n_parts=8).to(x.device)
                baselines = torch.zeros_like(x).to(x.device)

                amap = method.attribute(
                        x_device,
                        feature_mask=feature_mask,
                        target=used_target,
                        )                
            else:
                amap = method.attribute(x_device, target=used_target, **hparams)

            if torch.is_tensor(amap):
                amap = amap.squeeze().detach().cpu().numpy()
            else:
                amap = np.squeeze(np.asarray(amap))

            if np.sum(np.abs(amap)) == 0 or np.ptp(amap) == 0:
                print(f"[WARN] {mname} produced zero-valued map — injecting epsilon.")
                amap += 1e-6

            amap = (amap - np.min(amap)) / (np.ptp(amap) + 1e-8)
            attr_dict[mname] = amap
            print(f"[OK] {mname} done, shape={amap.shape}, sum={amap.sum():.3e}")
        except Exception as e:
            print(f"[WARN] XAI method {mname} failed: {e}")
            continue

    # Build cache
    cache: Dict[str, Dict[str, Any]] = {}
    x_cpu = x.detach().cpu().numpy()
    for mname, amap in attr_dict.items():
        cache[mname] = {"input": x_cpu.squeeze(), "amap": amap}

    # Initial visualization (3D overlay)
    initial_imgs = []
    try:
        default_alpha = heat_alpha
        default_channel = "Mean (all)"
        default_view = "axial"

        for mname, amap in attr_dict.items():
            entry = cache[mname]
            vol = entry["input"]
            amap_ = entry["amap"]

            if isinstance(vol, np.ndarray) and vol.ndim >= 3:
                default_slice = vol.shape[-3] // 2
            else:
                default_slice = 0

            img = overlay_heatmap_3d(
                vol,
                amap_,
                view=default_view,
                slice_idx=default_slice,
                alpha=default_alpha,
                channel_mode=default_channel,
            )
            img = add_label_to_image(img, mname)
            initial_imgs.append(img)
    except Exception as e:
        print(f"[WARN] Could not create initial overlay preview: {e}")

    preds_json = json.dumps({"top5": preds, "used_target": used_target}, indent=2)
    return initial_imgs, cache, preds_json


# Average time guesses (just for progress bar estimate)
AVG_METHOD_TIME = {
    "saliency": 0.3,
    "integratedgradients": 2.0,
    "inputxgradient": 0.5,
    "guidedbackprop": 1.0,
    "deeplift": 1.2,
    "deepliftshap": 2.5,
    "gradientshap": 2.5,
    "gradcam": 1.0,
    "occlusion": 5.0,
    "featureablation": 4.0,
    "shapleyvaluesampling": 6.0,
}


def run_xai_batch(
    model_state: Dict[str, Any],
    data_files: List[str],
    is_nifti: bool,
    slice_idx: int,
    methods: List[str],
    target_class: int,
    heat_alpha: float,
    progress=gr.Progress(),
):
    """
    Run XAI on a batch of files (possibly including zip).
    Returns:
      - all preview images
      - combined_cache (nested per-file)
      - all_preds JSON
    """
    import zipfile
    import tempfile

    if model_state is None or "_obj" not in model_state:
        raise gr.Error("Load or create a model first.")
    if not data_files:
        raise gr.Error("Please upload one or more images / volumes / .npz / .zip")

    # Expand zips
    expanded_files: List[str] = []
    for f in data_files:
        if f.lower().endswith(".zip"):
            with zipfile.ZipFile(f, "r") as zf:
                tmpdir = tempfile.mkdtemp()
                zf.extractall(tmpdir)
                expanded_files.extend([str(p) for p in Path(tmpdir).rglob("*") if p.is_file()])
        else:
            expanded_files.append(f)

    total_jobs = len(expanded_files) * max(len(methods), 1)

    rough_estimate = 0.0
    for m in methods:
        rough_estimate += AVG_METHOD_TIME.get(m.lower(), 1.0)
    rough_estimate *= len(expanded_files)
    progress(0, desc=f"Estimated total time ~{rough_estimate:.1f}s")

    all_results: List[Any] = []
    combined_cache: Dict[str, Dict[str, Any]] = {}
    all_preds: List[Any] = []

    start_time = time.time()
    job_count = 0

    #for f, _ in itertools.product(expanded_files, methods if methods else [None]):
    # NEW (CORRECT)
    for f in expanded_files:    
        try:
            imgs, cache, preds_json = run_xai_single(
                model_state,
                f,
                is_nifti,
                slice_idx,
                methods,
                target_class,
                heat_alpha,
            )
            all_results.extend(imgs)

            if f not in combined_cache:
                combined_cache[f] = {}
            combined_cache[f].update(cache)

            all_preds.append({f: json.loads(preds_json)})
        except Exception as e:
            err_img = make_error_image(str(e))
            all_results.append(err_img)

        job_count += 1
        elapsed = time.time() - start_time
        avg_per_job = elapsed / max(job_count, 1)
        remaining = (total_jobs - job_count) * avg_per_job
        progress(job_count / total_jobs, desc=f"Remaining ~{remaining:.1f}s")

    return all_results, combined_cache, json.dumps(all_preds, indent=2)
