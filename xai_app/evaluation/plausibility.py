# xai_app/evaluation/plausibility.py

import os
import numpy as np
from scipy.ndimage import center_of_mass


# ==========================================================
# LOAD GTV MASKS (matching your original logic)
# ==========================================================
def load_npz_gtv(npz_path: str) -> np.ndarray:
    """
    Load an NPZ file with arr_0[2] = GTV.
    This EXACTLY matches your original code.
    """
    data = np.load(npz_path)
    arr = data["arr_0"] if "arr_0" in data else list(data.values())[0]

    if arr.ndim != 4 or arr.shape[0] < 3:
        raise ValueError(f"Unexpected NPZ input shape: {arr.shape}")

    gtv = arr[2]                   # channel 2 = GTV
    gtv_mask = (gtv > 0).astype(bool)
    return gtv_mask


def build_pid_to_gtv_mask(gt_folder: str) -> dict:
    """
    Build mapping:
        'CHUM-001' --> CHUM-001_input.npz GTV mask
    """
    mapping = {}
    for fname in os.listdir(gt_folder):
        if not fname.endswith(".npz"):
            continue

        # expected pattern: CHUM-001_input.npz
        pid = fname.replace("_input.npz", "")

        full_path = os.path.join(gt_folder, fname)
        gtv_mask = load_npz_gtv(full_path)
        mapping[pid] = gtv_mask

    if len(mapping) == 0:
        raise RuntimeError(f"No NPZ masks found in {gt_folder}")

    print(f"[INFO] Loaded {len(mapping)} GTV masks.")
    return mapping


# ==========================================================
# TOP-K selection (exactly your old behaviour)
# ==========================================================
def get_topk_mask(values: np.ndarray, k_frac: float) -> np.ndarray:
    v = values.ravel()
    N = v.size
    if k_frac <= 0 or N == 0:
        return np.zeros_like(values, dtype=bool)
    if k_frac >= 1:
        return np.ones_like(values, dtype=bool)
    k = max(1, int(round(k_frac * N)))
    idx = np.argpartition(v, N - k)[-k:]
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    return mask.reshape(values.shape)


# ==========================================================
# METRICS (exact copy of your code)
# ==========================================================
def dice_coef(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    a = a_mask.astype(bool)
    b = b_mask.astype(bool)
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return (2.0 * inter) / (a.sum() + b.sum() + 1e-8)


def iou_coef(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    a = a_mask.astype(bool)
    b = b_mask.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return inter / (union + 1e-8)


def pointing_game(amap_pos: np.ndarray, gtv_mask: np.ndarray) -> float:
    if amap_pos.size == 0 or np.all(amap_pos <= 0):
        return 0.0
    max_idx = np.unravel_index(np.argmax(amap_pos), amap_pos.shape)
    return 1.0 if gtv_mask[max_idx] else 0.0


def precision_recall_at_k(amap_pos: np.ndarray, gtv_mask: np.ndarray, k_frac: float):
    sel = get_topk_mask(amap_pos, k_frac)
    sel_n = sel.sum()
    gtv_n = gtv_mask.sum()
    tp = np.logical_and(sel, gtv_mask).sum()
    precision = (tp / (sel_n + 1e-8)) if sel_n > 0 else 0.0
    recall = (tp / (gtv_n + 1e-8)) if gtv_n > 0 else 0.0
    return float(precision), float(recall)


def anatomical_plausibility_index(amap_pos: np.ndarray, gtv_mask: np.ndarray) -> float:
    if gtv_mask.sum() == 0:
        return np.nan
    if np.all(amap_pos <= 0):
        return np.nan
    c_gtv = np.array(center_of_mass(gtv_mask))
    c_sal = np.array(center_of_mass(amap_pos))
    dist = np.linalg.norm(c_gtv - c_sal)
    diag = np.linalg.norm(amap_pos.shape)
    api = 1.0 - dist / (diag + 1e-8)
    return max(api, 0.0)

# ==========================================================
# K–based aggregated metrics (use 1 or more k values)
# ==========================================================

# Default list, matching your old script: [0.01, 0.005]
DEFAULT_K_LIST = (0.01, 0.005)


def _normalize_k_list(k_list):
    """
    Helper: make sure k_list is an iterable of floats.
    If None -> use DEFAULT_K_LIST.
    If single float -> wrap to [k].
    """
    if k_list is None:
        return list(DEFAULT_K_LIST)
    if isinstance(k_list, (float, int)):
        return [float(k_list)]
    return [float(k) for k in k_list]


def dice_k(amap_pos: np.ndarray, gtv_mask: np.ndarray, k_list=None) -> float:
    """
    Mean Dice over all k in k_list.
    Equivalent to averaging Dice@0.01 and Dice@0.005 if you use DEFAULT_K_LIST.
    """
    ks = _normalize_k_list(k_list)
    vals = []
    for k in ks:
        pred_mask = get_topk_mask(amap_pos, k)
        vals.append(dice_coef(pred_mask, gtv_mask))
    return float(np.mean(vals))


def iou_k(amap_pos: np.ndarray, gtv_mask: np.ndarray, k_list=None) -> float:
    """
    Mean IoU over all k in k_list.
    """
    ks = _normalize_k_list(k_list)
    vals = []
    for k in ks:
        pred_mask = get_topk_mask(amap_pos, k)
        vals.append(iou_coef(pred_mask, gtv_mask))
    return float(np.mean(vals))


def precision_k(amap_pos: np.ndarray, gtv_mask: np.ndarray, k_list=None) -> float:
    """
    Mean Precision@k over all k in k_list.
    """
    ks = _normalize_k_list(k_list)
    vals = []
    for k in ks:
        p, _ = precision_recall_at_k(amap_pos, gtv_mask, k)
        vals.append(p)
    return float(np.mean(vals))


def recall_k(amap_pos: np.ndarray, gtv_mask: np.ndarray, k_list=None) -> float:
    """
    Mean Recall@k over all k in k_list.
    """
    ks = _normalize_k_list(k_list)
    vals = []
    for k in ks:
        _, r = precision_recall_at_k(amap_pos, gtv_mask, k)
        vals.append(r)
    return float(np.mean(vals))
