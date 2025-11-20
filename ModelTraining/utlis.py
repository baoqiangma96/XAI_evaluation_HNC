import pandas as pd
import numpy as np
import torch
import os
import monai
import matplotlib.pyplot  as plt
import random
import torch.nn as nn

# sampler of oversampling the training set
def get_oversampler(endpoint_info, train_ID, event_name):
        label_raw_train = np.array(list(endpoint_info.loc[endpoint_info['PatientID'].isin(train_ID)][event_name]))
        weights = 1/ np.array([np.count_nonzero(1 - label_raw_train), np.count_nonzero(label_raw_train)]) # check event and no events samples numbers
        samples_weight = np.array([weights[t] for t in label_raw_train])
        samples_weight = torch.from_numpy(samples_weight) 
        sampler  = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler
    
def seed_torch(seed=42,deter=False):
    # `deter` means use deterministic algorithms for GPU training reproducibility, 
    #if set `deter=True`, please set the environment variable `CUBLAS_WORKSPACE_CONFIG` in advance
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    from monai.utils import set_determinism
    set_determinism(seed=seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def get_data_dict_hecktor(patientsID , opt, endpoint_info): 
    
    
    data_dict = []
    for i , pID in enumerate(patientsID):
        
        data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }      
                
        event  = np.array(endpoint_info.loc[endpoint_info['PatientID'] == pID][opt.event_name])[0].astype(float)
        event_time  = np.array(endpoint_info.loc[endpoint_info['PatientID'] == pID][opt.event_time_name])[0].astype(float)
        
        data_single_dict[opt.event_name]=  torch.tensor(event,dtype = torch.float)
        data_single_dict[opt.event_time_name]=  torch.tensor(event_time,dtype = torch.float)
        
        # ? Add PatientID explicitly
        data_single_dict["PatientID"] = pID
        
        data_dict.append(data_single_dict)
    return data_dict       
    
    
def get_model(opt): 
    # ? Determine number of input channels
    if getattr(opt, "sum_channel", False):
        # When summing modalities ? 1 combined channel
        input_channel = 1
    else:
        # When concatenating ? one channel per modality
        input_channel = len(opt.input_modality)

    if opt.model == 'ResNet18':
        return monai.networks.nets.resnet18(
            spatial_dims=3,
            n_input_channels=input_channel,
            num_classes=1,
            conv1_t_stride=2
        )

    elif opt.model == 'DenseNet121':
        return monai.networks.nets.DenseNet121(
            spatial_dims=3,
            in_channels=input_channel,
            out_channels=1
        )

    else:
        raise ValueError(f"? Unsupported model name: {opt.model}")

   