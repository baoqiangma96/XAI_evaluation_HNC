# xai_app/io/volume_loader.py
from typing import Union
import numpy as np
from PIL import Image
import torch

try:
    import SimpleITK as sitk
    HAS_SITK = True
except Exception:
    HAS_SITK = False

try:
    import nibabel as nib
    HAS_NIB = True
except Exception:
    HAS_NIB = False


def load_nifti_slice(path: str, slice_index: int) -> Image.Image:
    vol = None
    if HAS_SITK:
        img = sitk.ReadImage(path)
        vol = sitk.GetArrayFromImage(img)
    elif HAS_NIB:
        img = nib.load(path)
        vol = img.get_fdata()
        if vol.ndim == 4:
            vol = vol[..., 0]
        vol = np.transpose(vol, (2, 1, 0))
    else:
        raise RuntimeError("Install SimpleITK or nibabel to read NIfTI.")

    D = vol.shape[0]
    s = int(np.clip(slice_index, 0, D - 1))
    sl = vol[s].astype(np.float32)
    if np.ptp(sl) > 0:
        sl = (sl - sl.min()) / (sl.max() - sl.min())
    pil = Image.fromarray((sl * 255).astype(np.uint8), mode="L").convert("RGB")
    return pil


def load_npy_volume(path: str, device: str = "cpu") -> torch.Tensor:
    """
    Load preprocessed 3D npz/npy file into shape (1, C, D, H, W).
    """
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr["x"]

    if arr.ndim == 4:
        arr = np.mean(arr, axis=0, keepdims=True)  # (1, D, H, W)
    elif arr.ndim == 3:
        arr = arr[np.newaxis, ...]  # (1, D, H, W)

    x = torch.from_numpy(arr).float().unsqueeze(0).to(device)  # (1, C, D, H, W)
    return x
