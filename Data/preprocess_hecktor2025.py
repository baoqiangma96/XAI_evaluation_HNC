#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HECTOR 2025 Preprocessing Pipeline
----------------------------------
1) Load clinical CSVs & find overlapping patients
2) Stratified train/test split
3) Preprocess CT/PT/GTV (crop → resample → save NIfTI + preview)
4) Convert to 3-channel NPZ (CT, PT, GTV for XAI models)

Author: Baoqiang Ma
Date: 2025-01-20
"""

import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, ResizeD,
    ScaleIntensityRangeD, ConcatItemsD, ToTensord
)
from monai.data import Dataset


# ------------------------------------------------
# Utility: resample SimpleITK image
# ------------------------------------------------
def resample_image(img, new_spacing, interpolator):
    spacing = np.array(img.GetSpacing())
    size = np.array(img.GetSize())
    new_size = np.round(size * (spacing / new_spacing)).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())

    return resampler.Execute(img)


# ------------------------------------------------
# Utility: centroid of GTV mask
# ------------------------------------------------
def get_center_of_mass(mask):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    labels = stats.GetLabels()

    if not labels:
        raise ValueError("Empty GTV mask.")

    centers = [stats.GetCentroid(l) for l in labels]
    return np.mean(np.array(centers), axis=0)


# ------------------------------------------------
# Crop 3D image around physical coordinate
# ------------------------------------------------
def crop_center(img, center_mm, crop_mm):
    spacing = np.array(img.GetSpacing())
    size = np.array(img.GetSize())

    center_idx = np.round(img.TransformPhysicalPointToIndex(center_mm)).astype(int)
    crop_size_vox = np.round(crop_mm / spacing).astype(int)

    start = (center_idx - crop_size_vox // 2).clip(0, size - crop_size_vox)
    end = start + crop_size_vox

    region = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
    arr = sitk.GetArrayFromImage(img)

    # SITK array is [z, y, x]
    cropped_arr = arr[region[2], region[1], region[0]]

    cropped = sitk.GetImageFromArray(cropped_arr)
    cropped.SetSpacing(tuple(spacing))
    cropped.SetDirection(img.GetDirection())

    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    start_phys = origin + direction @ (start * spacing)
    cropped.SetOrigin(tuple(start_phys))

    return cropped


# ------------------------------------------------
# Preview slice
# ------------------------------------------------
def save_preview(ct, pt, gtv, out_file):
    ct_arr = sitk.GetArrayFromImage(ct)
    pt_arr = sitk.GetArrayFromImage(pt)
    gtv_arr = sitk.GetArrayFromImage(gtv)

    mid = ct_arr.shape[0] // 2

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(ct_arr[mid], cmap="gray"); ax[0].set_title("CT")
    ax[1].imshow(pt_arr[mid], cmap="inferno"); ax[1].set_title("PT")
    ax[2].imshow(gtv_arr[mid], cmap="gray"); ax[2].set_title("GTV")

    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()


# ------------------------------------------------
# 1) FIND OVERLAP + STRATIFIED SPLIT
# ------------------------------------------------
def compute_overlap_and_split(task1_csv, task2_csv, out_csv):

    t1 = pd.read_csv(task1_csv)
    t2 = pd.read_csv(task2_csv)

    overlap = set(t1.PatientID) & set(t2.PatientID)
    print(f"Found {len(overlap)} overlapping patients")

    t2 = t2[t2.PatientID.isin(overlap)].copy()
    t2 = t2[["PatientID", "Relapse", "RFS"]]

    train_ids, test_ids = train_test_split(
        t2["PatientID"], test_size=0.25,
        stratify=t2["Relapse"], random_state=42
    )

    t2["Set"] = "train"
    t2.loc[t2.PatientID.isin(test_ids), "Set"] = "test"
    t2.to_csv(out_csv, index=False)

    print(f"Saved overlap split: {out_csv}")
    return t2


# ------------------------------------------------
# 2) PREPROCESS CT/PT/GTV → NIFTI
# ------------------------------------------------
def preprocess_nifti(df, task1_dir, out_dir, preview_dir,
                     crop_mm=np.array([192,192,192]),
                     spacing=np.array([2,2,2])):

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for pid in df.PatientID:
        folder = task1_dir / pid
        ct_path = folder / f"{pid}__CT.nii.gz"
        pt_path = folder / f"{pid}__PT.nii.gz"
        gtv_path = folder / f"{pid}.nii.gz"

        if not (ct_path.exists() and pt_path.exists() and gtv_path.exists()):
            print(f"[WARN] Missing files for {pid}, skipping.")
            continue

        ct = sitk.ReadImage(str(ct_path))
        pt = sitk.ReadImage(str(pt_path))
        gtv = sitk.ReadImage(str(gtv_path))

        # combine GTV labels 1 and 2
        gtv_arr = sitk.GetArrayFromImage(gtv)
        gtv_comb = sitk.GetImageFromArray(((gtv_arr==1)|(gtv_arr==2)).astype(np.uint8))
        gtv_comb.CopyInformation(gtv)

        # center
        center = get_center_of_mass(gtv_comb)

        ct_c = crop_center(ct, center, crop_mm)
        pt_c = crop_center(pt, center, crop_mm)
        gtv_c = crop_center(gtv_comb, center, crop_mm)

        ct_r = resample_image(ct_c, spacing, sitk.sitkLinear)
        pt_r = resample_image(pt_c, spacing, sitk.sitkLinear)
        gtv_r = resample_image(gtv_c, spacing, sitk.sitkNearestNeighbor)

        sitk.WriteImage(ct_r, str(out_dir / f"{pid}__CT.nii.gz"))
        sitk.WriteImage(pt_r, str(out_dir / f"{pid}__PT.nii.gz"))
        sitk.WriteImage(gtv_r, str(out_dir / f"{pid}__gtv.nii.gz"))

        save_preview(ct_r, pt_r, gtv_r, preview_dir / f"{pid}_preview.png")

        print(f"✓ Processed {pid}")


# ------------------------------------------------
# 3) CONVERT TO CT+PT+GTV NPZ
# ------------------------------------------------
def generate_npz(nifti_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]
    pids = sorted({re.sub(r"__(CT|PT|gtv)\.nii\.gz$", "", f) for f in files})

    data_dicts = []
    for pid in pids:
        entry = {
            "CT":  str(nifti_dir / f"{pid}__CT.nii.gz"),
            "PT":  str(nifti_dir / f"{pid}__PT.nii.gz"),
            "gtv": str(nifti_dir / f"{pid}__gtv.nii.gz"),
        }
        if not all(os.path.exists(v) for v in entry.values()):
            print(f"[WARN] Missing modality for {pid}, skipping.")
            continue
        data_dicts.append(entry)

    test_tf = Compose([
        LoadImageD(keys=["CT","PT","gtv"]),
        EnsureChannelFirstD(keys=["CT","PT","gtv"]),
        ResizeD(keys=["CT","PT","gtv"], spatial_size=(96,96,96),
                mode=("trilinear","trilinear","nearest")),
        ScaleIntensityRangeD("CT", a_min=-200, a_max=200, b_min=0, b_max=1, clip=True),
        ScaleIntensityRangeD("PT", a_min=0, a_max=25, b_min=0, b_max=1, clip=True),
        ConcatItemsD(keys=["CT","PT","gtv"], name="image"),
        ToTensord(keys="image"),
    ])

    ds = Dataset(data=data_dicts, transform=test_tf)

    for i, item in enumerate(tqdm(ds, desc="Generating npz")):
        pid = pids[i]
        arr = item["image"].numpy().astype(np.float32)
        np.savez_compressed(out_dir / f"{pid}_input.npz", x=arr)

    print(f"✓ Saved {len(ds)} NPZ files to {out_dir}")


# ------------------------------------------------
# CLI
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1_csv", required=True)
    parser.add_argument("--task2_csv", required=True)
    parser.add_argument("--task1_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_nifti", required=True)
    parser.add_argument("--out_preview", required=True)
    parser.add_argument("--out_npz", required=True)
    args = parser.parse_args()

    # --- NEW: create output directories if missing ---
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_nifti).mkdir(parents=True, exist_ok=True)
    Path(args.out_preview).mkdir(parents=True, exist_ok=True)
    Path(args.out_npz).mkdir(parents=True, exist_ok=True)

    # --- run pipeline ---
    df = compute_overlap_and_split(args.task1_csv, args.task2_csv, args.out_csv)
    preprocess_nifti(df, Path(args.task1_dir), Path(args.out_nifti), Path(args.out_preview))
    generate_npz(Path(args.out_nifti), Path(args.out_npz))



if __name__ == "__main__":
    main()
