# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:33:20 2022

@author: MaB
"""

import torch 
import torch.nn as nn
from lifelines.utils import concordance_index
import wandb

from scipy.ndimage import binary_dilation
  
def test_hecktor(model, test_loader, device, opt, endpoint_info, test_ID, mode='test'):
    model.eval()
    test_outputs_cat = torch.Tensor().to(device)

    with torch.no_grad():
        for test_data in test_loader:
            inputs_list = []

            # ---- Combine input modalities (same as in training)
            for image in opt.input_modality:
                sub_data = test_data[image].to(device)
                if image == 'gtv':
                    sub_data[sub_data > 0] = 1
                inputs_list.append(sub_data)

            if getattr(opt, "sum_channel", False):
                # ? Sum & average if enabled
                test_images = torch.stack(inputs_list, dim=0).sum(dim=0) / len(inputs_list)
            else:
                # ? Concatenate normally
                test_images = torch.cat(inputs_list, dim=1)

            # ---- Forward pass
            test_outputs = model(test_images)
            test_outputs_cat = torch.cat((test_outputs_cat, test_outputs), 0)

        # ---- Evaluation
        test_outputs_cat = test_outputs_cat.detach().cpu().numpy()
        epoch_score = -test_outputs_cat[:, 0]
        epoch_time = list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_time_name])
        epoch_event = list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_name])

        metric = concordance_index(epoch_time, epoch_score, epoch_event)
        print(f"{mode}_epoch C-index: {metric:.4f}")
        wandb.log({f"{mode}_epoch_c-index": metric})

    return metric
