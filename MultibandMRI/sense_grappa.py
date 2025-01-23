import torch 
from torch import Tensor
from typing import Tuple 
from MultibandMRI import get_kernel_patches, get_kernel_points, get_num_interpolated_points, get_kernel_shifts, interp_to_matrix_size, ifft1d, fft1d, fft2d, ifft2d

class sense_grappa:

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
        self.accel = (self.sms, accel[1])
        self.kernel_size = kernel_size 
        self.tik = tik 
        self.final_matrix_size = final_matrix_size
        self.calibrate(calib_data)

    def calibrate(self, data):

        # concatenate SMS data along readout dimension
        data = ifft1d(data, dim=2)
        data = torch.cat([data[None,s,...] for s in range(self.sms)], dim=2)
        data = fft1d(data, dim=2)

        # get the source data points 
        A = get_kernel_patches(data, kernel_size=self.kernel_size, accel=self.accel)
        self.kernel_shifts, self.start_inds, self.eff_kernel_size = get_kernel_shifts(self.kernel_size, self.accel) 

        # l2 regularization 
        AH = A.conj().transpose(2,3)
        _,S,_ = torch.linalg.svd(A, full_matrices=False)
        vals = torch.max(torch.abs(S), dim=-1).values
        lamda = self.tik * vals[:,:,None,None]
        I = torch.eye(AH.shape[2], dtype=A.dtype, device=A.device)[None,None,:,:]
        AHA_inv = torch.linalg.inv(AH@A + lamda*I)

        # calculate the weights for each offset relative to "top left" kernel
        # point (i.e., to account for in-plane acceleration)
        self.weights = []
        for shifts in self.kernel_shifts:
            b = get_kernel_points(data, shifts=shifts, kernel_size=self.kernel_size, accel=self.accel)
            self.weights.append(AHA_inv @ (AH @ b))


    def apply(self, inp_data):

        # readout FOV of extended-FOV images is no longer centered for an even number of simultaneously excited slices. add FOV/2 shift here
        if self.sms % 2 == 0: inp_data[:,:,::2,:] = inp_data[:,:,::2,:] * torch.exp(torch.tensor([1j], dtype=torch.complex64, device=inp_data.device))

        # zero-fill data 
        data = torch.zeros((inp_data.shape[0], inp_data.shape[1], self.sms*inp_data.shape[2], inp_data.shape[3]), dtype=inp_data.dtype, device=inp_data.device)
        data[:,:,::self.sms,:] = inp_data 

        # figure out number of interpolated points along each dimension 
        nr, nc = get_num_interpolated_points(data.shape, self.kernel_size, self.accel)

        # interpolate the missing points
        A = get_kernel_patches(data, kernel_size=self.kernel_size, accel=self.accel, stride=self.accel)
        Y = [(A@w).view(1, self.coils, nr, nc) for w in self.weights]
        out = torch.zeros((1, self.coils, self.accel[0]*nr, self.accel[1]*nc), dtype=inp_data.dtype, device=inp_data.device)
        for rfe, rpe in self.start_inds:
            out[:,:,rfe::self.accel[0],rpe::self.accel[1]] = Y[rfe*self.accel[1]+rpe]

        # final interpolation 
        if self.final_matrix_size is not None:
            adjusted_matrix_size = (self.sms*self.final_matrix_size[0], self.final_matrix_size[1])
            out = interp_to_matrix_size(out, adjusted_matrix_size)

        # data consistency
        out[torch.abs(data) > 0.0] = data[torch.abs(data) > 0.0]

        # bring to the image domain and crop slices
        nread = inp_data.shape[2]
        img = ifft2d(out, dims=(2,3))
        img = torch.stack([img[0,:,n*nread:(n+1*nread),:] for n in range(self.sms)], axis=0)
        slc_ksp = fft2d(img, dims=(2,3))
        rss = torch.sqrt(torch.sum(torch.abs(img * img.conj()), dim=1))

        return slc_ksp, rss
