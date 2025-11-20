# -*- coding: utf-8 -*-
'''
Created on Fri Jun 17 16:05:46 2022
@author: MaB
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--result_path',
        default='./result/',
        type=str,
        help='Result directory path')
    
    parser.add_argument(
        '--data_path',
        default='./Data/preprocessed/',
        type=str,
        help='Data directory path (after resampled)')

    parser.add_argument(
        '--endpoint_path',
        default='./Data/clinical_data/HECKTOR_2025_Training_Task_2_overlap_split_2yr.csv',
        type=str,
        help='Endpoint information path')

    parser.add_argument(
        '--learning_rate',
        default=2e-4,
        type=float,
        help=
        'Initial learning rate ')

    parser.add_argument('--momentum', default=0.90, type=float, help='Momentum')

    parser.add_argument(
        '--weight_decay', default=5e-5, type=float, help='Weight Decay')


    parser.add_argument(
        '--batch_size', 
        default= 12, 
        type=int, 
        help='Batch Size')


    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')


    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)

    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)

    parser.add_argument(
        '--no_test',
        action='store_true',
        help='If true, test is not performed.')
    parser.set_defaults(no_test=False)


    parser.add_argument(
        '--model',
        default='Densenet121',
        type=str,
        help='(resnet/ denset and all nets from MONAI )')

    parser.add_argument(
        '--model_actfn',
        default='relu',
        type=str,
        help='activation function')

    parser.add_argument(
        '--input_modality',
        nargs='+',
        help='Different types of input modality selection: CT, PT, gtv')

    parser.add_argument(
        '--fold',
        default=1,
        type=int,
        help='fold number')

    parser.add_argument(
        '--oversample',
        type= bool,
        default= False , 
        help='If true, oversample is performed.')
        
    parser.add_argument(
        '--sum_channel',
        type= bool,
        default= False , 
        help='If true, sum channel is performed.')
        

    args = parser.parse_args()

    return args
