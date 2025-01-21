import torch 
from torch import Tensor
from typing import Tuple 
# from MultibandMRI import extract_and_flatten_multicoil_kspace_patches, extract_point_within_multicoil_kspace_patch
from MultibandMRI import get_kernel_patches, get_kernel_points

class slice_grappa:

    def __init__(self,
                 calib_data: Tensor,
                 accel: Tuple=(1,1),
                 kernel_size: Tuple=(3,3),
                 tik: float=0.0):
        '''
        Input:
            calib_data: (sms, coils, readout, phase) complex64 tensor
            phase_accel: integer (uniform) acceleration factor 
            kernel_size: tuple of (readout, phase) kernel sizes
        '''

        self.sms, self.coils, _, _ = calib_data.shape
        self.accel = accel
        self.kernel_size = kernel_size 
        self.tik = tik 
        self.calibrate(calib_data)

    def calibrate(self, calib_data):
        
        # "source" data for slice grappa calibration is the multiband k-space 
        source = torch.sum(calib_data, dim=0, keepdim=True)

        print(source.shape)
        A = get_kernel_patches(source, kernel_size=self.kernel_size, accel=self.accel)

        # if l2 regularization is desired 
        # if self.tik > 0.0:
        AH = A.conj().transpose(2,3)
        I = torch.eye(AH.shape[0], dtype=A.dtype, device=A.device)[None,None,:,:]
        AHA_inv = torch.linalg.inv(AH@A + self.tik*I)

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
        nread = data.shape[2]
        A = get_kernel_patches(data, kernel_size=self.kernel_size, accel=self.accel, pad=True)
        Y = [(A@w).view(self.sms, self.coils, nread, -1) for w in self.weights]
        out = torch.zeros_like(Y[0])
        for rfe in range(self.accel[0]):
            for rpe in range(self.accel[1]):
                out[:,:,rfe::self.accel[0],rpe::self.accel[1]] = Y[rfe*self.accel[1]+rpe][:,:,rfe::self.accel[0],rpe::self.accel[1]]
        # out = torch.zeros((self.sms, self.coils, nread, ny), dtype=data.dtype, device=data.device)
        # for r in range(self.phase_accel):
        #     out[:,:,:,r::self.phase_accel] = Y[r]
        return Y[0]





# class slice_grappa:

#     def __init__(self,
#                  calib_data: Tensor,
#                  phase_accel: int=1,
#                  kernel_size: Tuple=(3,3),
#                  tik: float=0.0):
#         '''
#         Input:
#             calib_data: (sms, coils, readout, phase) complex64 tensor
#             phase_accel: integer (uniform) acceleration factor 
#             kernel_size: tuple of (readout, phase) kernel sizes
#         '''

#         self.sms, self.coils, _, _ = calib_data.shape
#         self.phase_accel = int(phase_accel) 
#         self.kernel_size = kernel_size 
#         self.stride = (1, self.phase_accel)
#         self.tik = tik 
#         self.calibrate(calib_data)

#     def calibrate(self, calib_data):

#         source = torch.sum(calib_data, dim=0, keepdim=True)
#         A = extract_and_flatten_multicoil_kspace_patches(source, kernel_size=self.kernel_size, stride=self.stride)[:,None,:,:]
#         if self.tik > 0.0:
#             AH = A.conj().transpose(2,3)
#             I = torch.eye(AH.shape[0], dtype=A.dtype, device=A.device)[None,None,:,:]
#             AHA = AH@A + self.tik*I 
#         self.weights = []
#         fe_shift = self.kernel_size[0] // 2
#         pe_shift = (self.kernel_size[0]-1)*self.phase_accel // 2 - 1
#         for r in range(self.phase_accel):
#             shift = (fe_shift, pe_shift+r)
#             print(shift)
#             b = extract_point_within_multicoil_kspace_patch(calib_data, kernel_size=self.kernel_size, stride=self.stride, shift=shift)[...,None]
#             if self.tik > 0.0:
#                 X = torch.linalg.inv(AHA) @ (AH @ b)
#             else:
#                 X = torch.linalg.lstsq(A, b, rcond=None)[0]
#             self.weights.append(X)

#     def apply(self, data):
#         '''
#         Input:  
#             data: zero-filled measured SMS data (1, coils, readout, phase) complex64 tensor
#         '''

#         # nread = data.shape[2]
#         # A = extract_and_flatten_multicoil_kspace_patches(data, kernel_size=self.kernel_size, stride=self.stride, pad=True)[:,None,:,:]
#         # Y = []
#         # for r in range(self.phase_accel):
#         #     y = A @ self.weights[r]
#         #     y = y.view(self.sms, self.coils, nread, -1)
#         #     Y.append(y) 
#         # ny = sum([Y[r].shape[-1] for r in range(self.phase_accel)])
#         # out = torch.zeros((self.sms, self.coils, nread, ny), dtype=data.dtype, device=data.device)
#         # for r in range(self.phase_accel):
#         #     out[:,:,:,r::self.phase_accel] = Y[r]
#         # return out 

#         nread = data.shape[2]
#         A = extract_and_flatten_multicoil_kspace_patches(data, kernel_size=self.kernel_size, stride=(1,1), pad=True)[:,None,:,:]
#         Y = []
#         for r in range(self.phase_accel):
#             y = A @ self.weights[r]
#             y = y.view(self.sms, self.coils, nread, -1)
#             Y.append(y) 
#         ny = sum([Y[r].shape[-1] for r in range(self.phase_accel)])
#         out = torch.zeros((self.sms, self.coils, nread, ny), dtype=data.dtype, device=data.device)
#         for r in range(self.phase_accel):
#             out[:,:,:,r::self.phase_accel] = Y[r]
#         return out 
