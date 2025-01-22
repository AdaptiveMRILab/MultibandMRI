import torch 
from torch import Tensor
from typing import Tuple 
from MultibandMRI import get_kernel_patches, get_kernel_points

class split_slice_grappa:

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
        #source = torch.sum(calib_data, dim=0, keepdim=True)
        A = get_kernel_patches(calib_data, kernel_size=self.kernel_size, accel=self.accel)
        print(A.shape)

        # l2 regularization 
        AH = A.conj().transpose(2,3)
        _,S,_ = torch.linalg.svd(A, full_matrices=False)
        vals = torch.max(torch.abs(S), dim=-1).values
        lamda = (self.tik * vals[:,:,None,None])**2
        I = torch.eye(AH.shape[0], dtype=A.dtype, device=A.device)[None,None,:,:]
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
                self.weights.append(AHA_inv @ (AH @ b))

    def apply(self, data):

        # figure out number of interpolated points along each dimension 
        rows, cols = data.shape[2], data.shape[3]
        eff_row_kernel_size = (self.kernel_size[0] - 1) * self.accel[0] + 1
        eff_col_kernel_size = (self.kernel_size[1] - 1) * self.accel[1] + 1
        nr = rows - eff_row_kernel_size + 1
        nc = cols - eff_col_kernel_size + 1
        rv = torch.arange(0, nr, self.accel[0])
        cv = torch.arange(0, nc, self.accel[1])
        nr = torch.numel(rv)
        nc = torch.numel(cv)

        # interpolate the missing points
        A = get_kernel_patches(data, kernel_size=self.kernel_size, accel=self.accel, stride=self.accel)
        Y = [(A@w).view(self.sms, self.coils, nr, -1) for w in self.weights]
        out = torch.zeros_like(Y[0])
        for rfe in range(self.accel[0]):
            for rpe in range(self.accel[1]):
                out[:,:,rfe::self.accel[0],rpe::self.accel[1]] = Y[rfe*self.accel[1]+rpe][:,:,0::self.accel[0],0::self.accel[1]]
        
        # zero-fill to final matrix size 
        if self.final_matrix_size is not None:
            rowpad = self.final_matrix_size[0] - out.shape[2]
            rowpre = rowpad//2
            rowpst = rowpad - rowpre 
            colpad = self.final_matrix_size[1] - out.shape[3]
            colpre = colpad//2
            colpst = colpad - colpre 
            out = torch.nn.functional.pad(out, (colpre, colpst, rowpre, rowpst), mode='constant', value=0)
        
        return out
