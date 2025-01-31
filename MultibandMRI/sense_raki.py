import torch 
from torch import Tensor
from typing import Tuple 
import os 
from MultibandMRI import get_kernel_patches, get_kernel_points, get_kernel_shifts, get_num_interpolated_points, interp_to_matrix_size, ifft2d, train_complex_net, load_complex_net

class sense_raki:
    def __init__(self,
                calib_data: Tensor,
                recon_folder: str,
                accel: Tuple=(1,1),
                kernel_size: Tuple=(3,3),
                tik: float=0.0,
                final_matrix_size: Tuple=None,
                num_layers: int=4,
                hidden_size: int=128,
                num_epochs: int=100,
                random_seed: int=42,
                learn_rate: float=1e-4,
                train_split: float=0.75,
                scale_data: bool=False,
                loss_function: str='L1_L2',
                l2_frac: float=0.5,
                net_type: str='MLP',
                linear_weight: float=1.0):
        '''
        Input:
            calib_data: (sms, coils, readout, phase) complex64 tensor
            accel: integer (uniform) acceleration factors along frequency and phase dimensions
            kernel_size: tuple of (readout, phase) kernel sizes
            tik: l2 regularization parameter (scalar float)
            final_matrix_size: (final readout, final phase) tuple of output matrix sizes 
        '''

        self.sms, self.coils, _, _ = calib_data.shape
        self.accel = accel
        self.kernel_size = kernel_size 
        self.tik = tik 
        self.final_matrix_size = final_matrix_size
        self.linear_weight = linear_weight
        self.num_layers = num_layers 
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.learn_rate = learn_rate
        self.random_seed = random_seed
        self.recon_folder = recon_folder
        self.train_split = train_split
        self.scale_data = scale_data
        self.loss_function = loss_function
        self.net_type = net_type
        self.l2_frac = l2_frac
        # self.calibrate(calib_data) Commented out for initial commit