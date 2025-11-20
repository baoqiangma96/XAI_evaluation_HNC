# -*- coding: utf-8 -*-
"""
Simplified training script for TransRP
Author: MaB
Date: 2025-10-20
"""

import os
import torch
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from utlis import get_model, get_data_dict_hecktor, seed_torch, get_oversampler
from para_opts import parse_opts
from torch.utils.data import DataLoader
from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, Compose, ResizeD, ScaleIntensityRangeD,
    RandFlipD, RandAffineD, Rand3DElasticD
)
from monai.data import Dataset
import losses
from safetensors.torch import save_file, load_file
import wandb
from test import test_hecktor


# =============================
# Initialization
# =============================
seed_torch(42)


def main():
    # ---- Parse arguments and setup
    opt = parse_opts()
    '''
    # ✅ Add default flag (if missing from command line)
    if not hasattr(opt, "sum_channel"):
        opt.sum_channel = False  # default: concatenate
    '''

    wandb.init(project='ISBI2025', entity='mbq1137723824')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = torch.cuda.is_available()

    # ---- Event setup
    opt.event_name = 'Relapse'
    opt.event_time_name = 'RFS'

    # ---- Result path (show concat vs sum)
    comb_name = "sum" if opt.sum_channel else "concat"
    opt.result_path = (
        opt.result_path + f"{opt.model}_input_{opt.input_modality}_{comb_name}_OS_{opt.oversample}"
    )
    os.makedirs(opt.result_path, exist_ok=True)
    wandb.config.update(vars(opt), allow_val_change=True)

    # ---- Read endpoint info
    endpoint_info = pd.read_csv(opt.endpoint_path)
    train_ID = list(endpoint_info.loc[endpoint_info['Set'] == 'train', 'PatientID'])
    test_ID = list(endpoint_info.loc[endpoint_info['Set'] == 'test', 'PatientID'])

    # ---- Data dictionaries
    train_data_dict = get_data_dict_hecktor(train_ID, opt, endpoint_info.copy())
    test_data_dict = get_data_dict_hecktor(test_ID, opt, endpoint_info.copy())

    # =============================
    # Data transforms
    # =============================
    train_transforms = Compose([
        LoadImageD(keys=opt.input_modality),
        EnsureChannelFirstD(keys=opt.input_modality),
        ResizeD(
            keys=['CT', 'PT', 'gtv'],
            spatial_size=(96, 96, 96),
            mode=('trilinear', 'trilinear', 'nearest'),
            allow_missing_keys=True
        ),
        ScaleIntensityRangeD(keys='CT', a_min=-200, a_max=200, b_min=0, b_max=1, clip=True, allow_missing_keys=True),
        ScaleIntensityRangeD(keys='PT', a_min=0, a_max=25, b_min=0, b_max=1, clip=True, allow_missing_keys=True),
        RandFlipD(keys=opt.input_modality, prob=0.5, spatial_axis=[0, 1, 2], allow_missing_keys=True),
        RandAffineD(
            keys=['CT', 'PT', 'gtv'], prob=0.5,
            translate_range=(7, 7, 7),
            rotate_range=(np.pi/24,)*3,
            scale_range=(0.07,)*3,
            padding_mode='border',
            mode=('bilinear', 'bilinear', 'nearest'),
            allow_missing_keys=True
        ),
        Rand3DElasticD(
            keys=['CT', 'PT', 'gtv'], prob=0.2,
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            mode=('bilinear', 'bilinear', 'nearest'),
            allow_missing_keys=True
        )
    ])

    test_transforms = Compose([
        LoadImageD(keys=opt.input_modality),
        EnsureChannelFirstD(keys=opt.input_modality),
        ResizeD(
            keys=['CT', 'PT', 'gtv'],
            spatial_size=(96, 96, 96),
            mode=('trilinear', 'trilinear', 'nearest'),
            align_corners=(True, True, None),
            allow_missing_keys=True,
        ),
        ScaleIntensityRangeD(keys='CT', a_min=-200, a_max=200, b_min=0, b_max=1, clip=True, allow_missing_keys=True),
        ScaleIntensityRangeD(keys='PT', a_min=0, a_max=25, b_min=0, b_max=1, clip=True, allow_missing_keys=True),
    ])

    # =============================
    # DataLoaders
    # =============================
    num_workers = 10
    train_ds = Dataset(data=train_data_dict, transform=train_transforms)
    test_ds = Dataset(data=test_data_dict, transform=test_transforms)

    if opt.oversample:
        sampler = get_oversampler(endpoint_info, train_ID, opt.event_name)
        train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=num_workers,
                                  pin_memory=pin_memory, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_eval = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=pin_memory)

    # =============================
    # Model setup
    # =============================
    model = get_model(opt).to(device)
    criterion = losses.NegativeLogLikelihood()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.weight_decay
    )

    # =============================
    # Training loop
    # =============================
    max_epochs = 50
    print(f"🚀 Start training ({comb_name.upper()} mode) for {max_epochs} epochs using AdamW optimizer...")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_data in train_loader:
            optimizer.zero_grad()

            # ---- Combine input modalities
            inputs_list = []
            for image in opt.input_modality:
                sub_data = batch_data[image].to(device)
                if image == 'gtv':
                    sub_data[sub_data > 0] = 1
                inputs_list.append(sub_data)

            if opt.sum_channel:
                # ✅ Sum all modalities, then average
                inputs = torch.stack(inputs_list, dim=0).sum(dim=0) / len(inputs_list)
            else:
                # ✅ Concatenate along channel axis
                inputs = torch.cat(inputs_list, dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, batch_data[opt.event_time_name], batch_data[opt.event_name], model)
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss_step': loss.item()})
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch+1:03d}/{max_epochs}] - Train Loss: {epoch_loss:.4f}")
        wandb.log({'train_loss_epoch': epoch_loss, 'epoch': epoch+1})

        print("Evaluating train set...")
        test_hecktor(model, train_loader_eval, device, opt, endpoint_info, train_ID, mode='train')

        print("Evaluating test set...")
        test_hecktor(model, test_loader, device, opt, endpoint_info, test_ID, mode='test')

        # ---- Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(opt.result_path, f"epoch_{epoch+1}.safetensors")
            save_file(model.state_dict(), save_path)
            print(f"?? Saved model: {save_path}")

    print("✅ Training completed.")

    final_model_path = os.path.join(opt.result_path, "epoch_50.safetensors")
    if os.path.exists(final_model_path):
        print(f"?? Loading final weights from: {final_model_path}")
        state_dict = load_file(final_model_path)
        model.load_state_dict(state_dict)
        model.eval()

        print("📊 Evaluating final model on test set...")
        test_hecktor(model, test_loader, device, opt, endpoint_info, test_ID, mode='test')
    else:
        print("⚠️ Warning: final model file not found, skipping reload test.")

    print("🎯 All done! Results saved to:", opt.result_path)


if __name__ == '__main__':
    main()

