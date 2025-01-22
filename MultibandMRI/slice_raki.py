import torch 
from torch import Tensor
from typing import Tuple 
from MultibandMRI import get_kernel_patches, get_kernel_points, get_num_interpolated_points, interp_to_matrix_size

class slice_raki:

    def __init__(self,
                 calib_data: Tensor,
                 accel: Tuple=(1,1),
                 kernel_size: Tuple=(3,3),
                 tik: float=0.0,
                 final_matrix_size: Tuple=None):
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
        self.calibrate(calib_data)

    def calibrate(self, calib_data):

        
        
        # "source" data for slice grappa calibration is the multiband k-space 
        source = torch.sum(calib_data, dim=0, keepdim=True)
        A = get_kernel_patches(source, kernel_size=self.kernel_size, accel=self.accel)

        # l2 regularization 
        AH = A.conj().transpose(2,3)
        _,S,_ = torch.linalg.svd(A, full_matrices=False)
        vals = torch.max(torch.abs(S), dim=-1).values
        lamda = (self.tik * vals[:,:,None,None])**2
        I = torch.eye(AH.shape[2], dtype=A.dtype, device=A.device)[None,None,:,:]
        AHA_inv = torch.linalg.inv(AH@A + lamda*I)

        # calculate the weights for each offset relative to "top left" kernel
        # point (i.e., to account for in-plane acceleration)
        self.weights = []
        base_read_shift = (self.kernel_size[0] * self.accel[0])//2 
        base_phase_shift = (self.kernel_size[1] * self.accel[1])//2
        for rfe in range(self.accel[0]):
            for rpe in range(self.accel[1]):
                shifts = (base_read_shift+rfe, base_phase_shift+rpe)
                b = get_kernel_points(calib_data, shifts=shifts, kernel_size=self.kernel_size, accel=self.accel)
                # self.weights.append(AHA_inv @ (AH @ b))
                model = MLP()
                train_network(model, A, b, lr, num_epochs)

    def apply(self, data):

        # figure out number of interpolated points along each dimension 
        nr, nc = get_num_interpolated_points(data.shape, self.kernel_size, self.accel)

        # interpolate the missing points
        A = get_kernel_patches(data, kernel_size=self.kernel_size, accel=self.accel, stride=self.accel)
        Y = [(A@w).view(self.sms, self.coils, nr, -1) for w in self.weights]
        out = torch.zeros_like(Y[0])
        for rfe in range(self.accel[0]):
            for rpe in range(self.accel[1]):
                out[:,:,rfe::self.accel[0],rpe::self.accel[1]] = Y[rfe*self.accel[1]+rpe][:,:,0::self.accel[0],0::self.accel[1]]
        
        # zero-fill to final matrix size 
        if self.final_matrix_size is not None:
            out = interp_to_matrix_size(out, self.final_matrix_size)

        return out
