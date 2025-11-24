# xai_app/evaluation/run_eval.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple

import os
import numpy as np
import pandas as pd
import torch
import gradio as gr
import os
from pathlib import Path

from xai_app.utils.config_loader import load_latec_config
from xai_app.utils.paths import ensure_latec_on_syspath

ensure_latec_on_syspath()
from src.modules.xai_methods import XAIMethodsModule  # type: ignore
from src.modules.eval_metrics import EvalMetricsModule  # type: ignore


def clone_to_device(model: torch.nn.Module, device: str = "cpu") -> torch.nn.Module:
    import copy

    model = model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    new_model = copy.deepcopy(model).to(device)
    for m in new_model.modules():
        m._forward_hooks.clear()
        m._backward_hooks.clear()
        m._forward_pre_hooks.clear()
    return new_model.eval()


def run_eval_multi(
    model_state: Dict[str, Any],
    cache_dicts: Any,
    selected_latec_metrics: List[str],   # same
    selected_plausibility_metrics: List[str],   # NEW
    manual_target: int,
    preds_json_text: str,
    gt_file,
    is_nifti_mask: bool,
    slice_idx: int,
    progress=gr.Progress(),
):
    """
    Evaluate all selected metrics using generated attributions.
    Very close to your original run_eval_multi but moved to a module.
    """

    if model_state is None or not cache_dicts:
        raise gr.Error("Load a model and generate explanations first.")
    if not isinstance(cache_dicts, list):
        cache_dicts = [cache_dicts]

    # --- Modality detection from sample input ---
    modality = "image"
    sample_x_shape, sample_a_shape = None, None

    for cache in cache_dicts:
        if not cache:
            continue
        for _, methods_dict in cache.items():
            for _, entry in methods_dict.items():
                x_ = np.squeeze(np.asarray(entry.get("input")))
                a_ = np.squeeze(np.asarray(entry.get("amap")))
                sample_x_shape, sample_a_shape = x_.shape, a_.shape

                if len(sample_x_shape) == 4:
                    modality = "volume"
                elif len(sample_x_shape) == 3:
                    modality = "image" if sample_x_shape[0] in (1, 3) else "volume"
                else:
                    raise gr.Error(f"Unexpected input shape: {sample_x_shape}")
                break
            if sample_x_shape is not None:
                break
        if sample_x_shape is not None:
            break

    print(f"[INFO] Detected modality for evaluation: {modality} (x={sample_x_shape}, amap={sample_a_shape})")

    # --- Load configs and LATEC modules ---
    cfg_explain = load_latec_config(modality=modality, config_name="explain")
    cfg_eval = load_latec_config(modality=modality, config_name="eval")

    lm = model_state["_obj"]
    base_device = "cuda" if torch.cuda.is_available() else str(next(lm.model.parameters()).device)
    model = lm.model.to(base_device).eval()

    dummy_x = torch.zeros((1, 1, 96, 96, 96), device=base_device) if modality == "volume" else torch.zeros(
        (1, 3, 224, 224), device=base_device
    )

    xai_methods = XAIMethodsModule(cfg_explain, model, dummy_x)
    eval_module = EvalMetricsModule(cfg_eval, model)

    # Build XAI method lookup for metrics that need explain_func
    _method_lookup = {
        m.__class__.__name__.lower(): (m.attribute, params)
        for m, params in zip(xai_methods.xai_methods, xai_methods.xai_hparams)
    }

    if not selected_latec_metrics and not selected_plausibility_metrics:
        raise gr.Error("Please select at least one evaluation metric.")

    csv_partial = "xai_eval_partial_gpu.csv"
    if os.path.exists(csv_partial):
        df = pd.read_csv(csv_partial)
        print(f"[INFO] Loaded existing partial results ({len(df)} rows).")
    else:
        df = pd.DataFrame(columns=["Patient", "Method", "Metric", "Value"])
        print("[INFO] Starting new partial results file.")
        df.to_csv(csv_partial, index=False)

    total_jobs = sum(len(mdict) for cdict in cache_dicts for mdict in cdict.values())
    job_count = 0
    progress(0, desc=f"Evaluating {total_jobs} method maps...")

    batch_size = 41

    for cache_dict in cache_dicts:
        all_methods = {m for mdict in cache_dict.values() for m in mdict.keys()}
        for method_name in sorted(all_methods):
            patient_entries = [
                (pid, mdict[method_name])
                for pid, mdict in cache_dict.items()
                if method_name in mdict
            ]
            total_patients = len(patient_entries)
            print(f"[INFO] Evaluating {method_name} on {total_patients} patients (batch={batch_size})")

            for bstart in range(0, total_patients, batch_size):
                batch_entries = patient_entries[bstart:bstart + batch_size]
                x_list, y_list, a_list, pid_list = [], [], [], []

                for pid, entry in batch_entries:
                    x_np = np.squeeze(np.asarray(entry["input"]))
                    a_np = np.squeeze(np.asarray(entry["amap"]))

                    if np.sum(np.abs(a_np)) == 0 or np.ptp(a_np) == 0:
                        print(f"[WARN] {method_name} attribution map for {pid} all zeros — adding epsilon.")
                        a_np += 1e-6

                    if modality == "image":
                        if x_np.ndim == 3:
                            x_tensor = torch.tensor(x_np[None, ...], dtype=torch.float32)
                        else:
                            print(f"[WARN] Unexpected image shape {x_np.shape}, skipping")
                            continue
                    else:
                        if x_np.ndim == 4:
                            x_tensor = torch.tensor(x_np[None, ...], dtype=torch.float32)
                        elif x_np.ndim == 3:
                            x_tensor = torch.tensor(x_np[None, None, ...], dtype=torch.float32)
                        else:
                            print(f"[WARN] Unexpected volume shape {x_np.shape}, skipping")
                            continue

                    if a_np.ndim == x_np.ndim:
                        a_tensor = torch.tensor(a_np[None, None, ...], dtype=torch.float32)
                    elif a_np.ndim == x_np.ndim - 1:
                        a_tensor = torch.tensor(a_np[None, None, ...], dtype=torch.float32)
                    else:
                        print(f"[WARN] Unexpected amap shape {a_np.shape}, skipping")
                        continue

                    x_tensor = x_tensor.to(base_device)
                    a_tensor = a_tensor.to(base_device)
                    model = model.to(base_device).eval()

                    target_idx = 0
                    y_tensor = torch.tensor([target_idx], dtype=torch.long, device=base_device)

                    x_list.append(x_tensor)
                    a_list.append(a_tensor)
                    y_list.append(y_tensor)
                    pid_list.append(pid)

                if not x_list:
                    continue

                x_batch = torch.cat(x_list, dim=0)
                a_batch = torch.cat(a_list, dim=0)
                y_batch = torch.cat(y_list, dim=0)

                x_cpu = x_batch.detach().cpu().numpy()
                y_cpu = y_batch.detach().cpu().numpy()
                a_cpu = a_batch.detach().cpu().numpy()

                explain_func, explain_func_kwargs = None, None
                if method_name.lower() in _method_lookup:
                    explain_func, explain_func_kwargs = _method_lookup[method_name.lower()]
                    if isinstance(explain_func_kwargs, dict):
                        explain_func_kwargs = dict(explain_func_kwargs)

                if explain_func is not None:
                    if method_name.lower() in ["deepliftshap", "gradientshap"]:
                        x_mean = torch.mean(torch.tensor(x_cpu, device=base_device), dim=0)
                        n_baselines = 2
                        noise_std = 0.05 * (x_mean.max() - x_mean.min() + 1e-8)
                        baselines = x_mean.unsqueeze(0).repeat(n_baselines, 1, 1, 1, 1)
                        baselines += torch.randn_like(baselines) * noise_std
                        if explain_func_kwargs is None:
                            explain_func_kwargs = {}
                        explain_func_kwargs["baselines"] = baselines

                if method_name.lower() in ["lime", "kernelshap"]:
                    def _feature_mask_3d_divided(x_np, n_parts=8):
                        shape = x_np.shape
                        if len(shape) == 5:
                            _, c, d, h, w = shape
                        elif len(shape) == 4:
                            c, d, h, w = 1, *shape[1:]
                        else:
                            raise ValueError(f"Unexpected input shape for feature mask: {shape}")
                        pd, ph, pw = d // n_parts, h // n_parts, w // n_parts
                        mask = torch.zeros((1, c, d, h, w), dtype=torch.long)
                        idx = 0
                        for z in range(0, d, pd):
                            for y in range(0, h, ph):
                                for x_ in range(0, w, pw):
                                    mask[:, :, z:z+pd, y:y+ph, x_:x_+pw] = idx
                                    idx += 1
                        return mask
                    x_for_mask = x_cpu if x_cpu.ndim == 5 else x_cpu[:, None, ...]
                    feature_mask = _feature_mask_3d_divided(x_for_mask, n_parts=8).to(base_device)
                    baselines = torch.zeros_like(torch.tensor(x_for_mask, device=base_device))
                    if explain_func_kwargs is None:
                        explain_func_kwargs = {}
                    explain_func_kwargs.update({"feature_mask": feature_mask, "baselines": baselines})

                selected = set(m.lower() for m in selected_latec_metrics)

                for metric_obj in eval_module.eval_metrics:
                    mname = metric_obj.__class__.__name__.lower()
                    if mname not in selected:
                        continue

                    if x_cpu.ndim == 5 and x_cpu.shape[1] == 1:
                        x_cpu = x_cpu[:, 0]
                    if a_cpu.ndim == 5 and a_cpu.shape[1] == 1:
                        a_cpu = a_cpu[:, 0]

                    kwargs_common = dict(
                        model=model,
                        x_batch=x_cpu,
                        y_batch=y_cpu,
                        a_batch=a_cpu,
                        device=base_device,
                        custom_batch=[x_cpu, y_cpu, a_cpu, list(range(x_cpu.shape[0]))],
                    )

                    if explain_func is not None:
                        kwargs_common.update(
                            explain_func=explain_func,
                            explain_func_kwargs=explain_func_kwargs,
                        )

                    try:
                        if torch.cuda.is_available():
                            model = model.to("cuda")
                            kwargs_common["device"] = "cuda"
                            kwargs_common["model"] = model
                        else:
                            kwargs_common["device"] = "cpu"

                        try:
                            score = metric_obj(**kwargs_common, softmax=False)
                        except TypeError:
                            score = metric_obj(**kwargs_common)

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and torch.cuda.is_available():
                            print(f"[WARN] {mname} ran out of GPU memory → retrying on CPU.")
                            torch.cuda.empty_cache()
                            try:
                                cpu_model = clone_to_device(lm.model, "cpu")
                                kwargs_common.update(model=cpu_model, device="cpu")
                                try:
                                    score = metric_obj(**kwargs_common, softmax=False)
                                except TypeError:
                                    score = metric_obj(**kwargs_common)
                                del cpu_model
                            except Exception as e2:
                                print(f"[ERROR] CPU fallback failed for {mname}: {e2}")
                                batch_rows = [
                                    {"Patient": pid, "Method": method_name, "Metric": mname, "Value": np.nan}
                                    for pid in pid_list
                                ]
                                df = pd.concat([df, pd.DataFrame(batch_rows)], ignore_index=True)
                                df.to_csv(csv_partial, index=False)
                                continue
                        else:
                            print(f"[WARN] Metric {mname} failed: {e}")
                            batch_rows = [
                                {"Patient": pid, "Method": method_name, "Metric": mname, "Value": np.nan}
                                for pid in pid_list
                            ]
                            df = pd.concat([df, pd.DataFrame(batch_rows)], ignore_index=True)
                            df.to_csv(csv_partial, index=False)
                            continue

                    if isinstance(score, dict):
                        flat_vals = []
                        for vv in score.values():
                            if isinstance(vv, (list, np.ndarray)):
                                flat_vals.extend(vv)
                            else:
                                flat_vals.append(vv)
                        score_arr = np.array(flat_vals)
                    elif isinstance(score, (list, np.ndarray)):
                        score_arr = np.array(score)
                    else:
                        score_arr = np.repeat(score, len(pid_list))

                    if len(score_arr) < len(pid_list):
                        score_arr = np.repeat(np.nanmean(score_arr), len(pid_list))

                    batch_rows = []
                    for pid, s in zip(pid_list, score_arr):
                        batch_rows.append(
                            {"Patient": pid, "Method": method_name, "Metric": mname, "Value": float(np.nanmean(s))}
                        )

                    df = pd.concat([df, pd.DataFrame(batch_rows)], ignore_index=True)
                    df.to_csv(csv_partial, index=False)
                    print(f"[INFO] Saved partial results ({len(df)} rows) → {csv_partial}")

                del x_batch, y_batch, a_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                job_count += len(pid_list)
                progress(job_count / total_jobs, desc=f"{job_count}/{total_jobs} done (batched)")

    csv_partial = "xai_eval_partial_gpu.csv"
    if len(df) == 0 and os.path.exists(csv_partial):
        df = pd.read_csv(csv_partial)
        print(f"[INFO] Reloaded partial results ({len(df)} rows) from {csv_partial}")

    # ================================================================
    # 🔹 SECOND STAGE: Compute plausibility metrics (Dice, IoU, PG, API)
    #     - Using arr_0[2] = GTV from NPZ files
    #     - Using k_list = [0.01, 0.005] and returning the mean
    #     - NO ZIP support; only direct NPZ files or a folder
    # ================================================================
    if selected_plausibility_metrics:

        from xai_app.evaluation.plausibility import (
            build_pid_to_gtv_mask,
            dice_k,
            iou_k,
            precision_k,
            recall_k,
            pointing_game,
            anatomical_plausibility_index,
            DEFAULT_K_LIST,
        )

        if gt_file is None:
            raise gr.Error(
                "You selected plausibility metrics, but no GTV NPZ folder/files were provided."
            )

        print(f"[INFO] Computing plausibility metrics: {selected_plausibility_metrics}")

        # ---- which k values to use ----
        K_LIST = DEFAULT_K_LIST  # (0.01, 0.005)

        # ============================================================
        # NPZ-only mask loading (supports ONE or MULTIPLE uploaded files)
        # ============================================================

        from xai_app.evaluation.plausibility import load_npz_gtv

        if gt_file is None:
            raise gr.Error("Please upload at least one NPZ mask file.")

        # Gradio returns:
        #   - single file → string OR dict with .name
        #   - multiple files → list of strings
        # Normalize into a clean list of strings
        if isinstance(gt_file, list):
            gt_files = [f if isinstance(f, str) else f.name for f in gt_file]
        else:
            gt_files = [gt_file if isinstance(gt_file, str) else gt_file.name]

        pid_to_gtv = {}

        for f in gt_files:
            fname = Path(f).name
            if not fname.endswith(".npz"):
                print(f"[WARN] Skipping non-NPZ file: {fname}")
                continue

            # extract PID: CHUM-001_input.npz → CHUM-001
            pid = fname.replace("_input.npz", "")

            try:
                pid_to_gtv[pid] = load_npz_gtv(f)
            except Exception as e:
                print(f"[ERROR] Failed to load {fname}: {e}")

        if len(pid_to_gtv) == 0:
            raise gr.Error("No valid NPZ GTV masks found in uploaded files.")


        rows_pl = []

        # =========================================================
        # Loop over all attribution maps same as LATEC stage-1
        # =========================================================
        for cache_dict in cache_dicts:
            for pid, methods_dict in cache_dict.items():

                # pid might be a path → extract patient ID
                pid_stem = Path(str(pid)).stem          # "CHUM-001_input"
                pid_base = pid_stem.replace("_input", "")  # "CHUM-001"

                gtv_mask = pid_to_gtv.get(pid_base)
                if gtv_mask is None:
                    print(f"[WARN] No GTV mask found for patient {pid_base}, skipping.")
                    continue

                for mname, entry in methods_dict.items():
                    amap = np.squeeze(np.asarray(entry["amap"]))
                    amap_pos = np.maximum(amap, 0.0)

                    for metric_name in selected_plausibility_metrics:

                        if metric_name == "dice":
                            val = dice_k(amap_pos, gtv_mask, k_list=K_LIST)

                        elif metric_name == "iou":
                            val = iou_k(amap_pos, gtv_mask, k_list=K_LIST)

                        elif metric_name == "precision_at_k":
                            val = precision_k(amap_pos, gtv_mask, k_list=K_LIST)

                        elif metric_name == "recall_at_k":
                            val = recall_k(amap_pos, gtv_mask, k_list=K_LIST)

                        elif metric_name == "pointing_game":
                            val = pointing_game(amap_pos, gtv_mask)

                        elif metric_name == "api":
                            val = anatomical_plausibility_index(amap_pos, gtv_mask)

                        else:
                            continue

                        rows_pl.append(
                            {
                                "Patient": pid_base,
                                "Method": mname,
                                "Metric": metric_name,
                                "Value": float(val),
                            }
                        )

        # Save results into df
        if rows_pl:
            df = pd.concat([df, pd.DataFrame(rows_pl)], ignore_index=True)
            df.to_csv(csv_partial, index=False)
            print(f"[INFO] Added {len(rows_pl)} plausibility metric results.")
        else:
            print("[INFO] No plausibility metrics computed.")

   
   
    if len(df) == 0:
        raise gr.Error("No evaluation results produced. Check your metric definitions or cache content.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = (
        df.groupby(["Method", "Metric"])[numeric_cols]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.values]

    csv_summary = "xai_eval_summary_gpu.csv"
    csv_all = "xai_eval_all_gpu.csv"
    df.to_csv(csv_all, index=False)
    summary.to_csv(csv_summary, index=False)

    print(f"[INFO] ✅ Evaluation completed. Saved summary → {csv_summary}, all results → {csv_all}")
    return summary, csv_summary, csv_all 
