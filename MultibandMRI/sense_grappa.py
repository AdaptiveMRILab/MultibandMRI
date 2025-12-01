import torch 
from torch import Tensor
from typing import Tuple 
import numpy as np 
from MultibandMRI import get_kernel_patches, get_kernel_points, get_num_interpolated_points, get_kernel_shifts, interp_to_matrix_size, ifft1d, fft1d, fft2d, ifft2d
import matplotlib.pyplot as plt

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

    def apply(self,
              inp_data,
              noise_cov: Tensor=None,
              return_gfactor: bool=False):
        '''
        Args:
            inp_data: undersampled multiband k-space (1, coils, readout, phase)
            noise_cov: optional (coils x coils) noise covariance; defaults to identity
            sampling_density: optional 1D/phase vector for variable-density undersampling
            return_gfactor: if True, also return analytical g-factor map (rss domain)
        '''

        # readout FOV of extended-FOV images is no longer centered for an even number of simultaneously excited slices. add FOV/2 shift here
        if self.sms % 2 == 0: inp_data[:,:,1::2,:] = inp_data[:,:,1::2,:] * np.exp(1j*np.pi)

        # handling matrix sizes not evenly divisible by acceleration factor 
        phase_matrix_size = inp_data.shape[3]
        if inp_data.shape[3] % self.accel[1]:
            npad = self.accel[1] - (inp_data.shape[3] % self.accel[1])
            z = torch.zeros((inp_data.shape[0],inp_data.shape[1],inp_data.shape[2],npad), dtype=inp_data.dtype, device=inp_data.device)
            inp_data = torch.cat([inp_data, z], dim=3)

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

        # remove any extra zero padding lines that were added above
        data = data[...,:phase_matrix_size]

        # bring to the image domain and crop slices
        nread = inp_data.shape[2]
        img = ifft2d(out, dims=(2,3))
        img = torch.stack([img[0,:,n*nread:(n+1)*nread,:] for n in range(self.sms)], axis=0)
        slc_ksp = fft2d(img, dims=(2,3))
        rss = torch.sqrt(torch.sum(torch.abs(img * img.conj()), dim=1))

        if not return_gfactor:
            return slc_ksp, rss
        
        # get composite convolution kernel 
        conv_kernels = torch.stack([ torch.reshape(w, (self.sms, self.coils, self.coils) + self.kernel_size) for w in self.weights], dim=-1)
        conv_kernel = torch.zeros((self.sms, self.coils, self.coils, self.accel[0]*self.kernel_size[0], self.accel[1]*self.kernel_size[1]), dtype=conv_kernels.dtype, device=data.device)
        for rfe in range(self.accel[0]):
            for rpe in range(self.accel[1]):
                conv_kernel[..., rfe::self.accel[0], rpe::self.accel[1]] = torch.flip(conv_kernels[..., rfe*self.accel[1]+rpe], dims=(-1,-2))
        
        # zero-pad the kernel and bring it to the image domain 
        conv_kernel_pad = torch.zeros((1, self.coils, self.coils, self.sms*inp_data.shape[2], inp_data.shape[3]), dtype=conv_kernel.dtype, device=data.device)
        r1 = int(conv_kernel_pad.shape[3]//2 - conv_kernel.shape[-2]//2) 
        r2 = r1 + conv_kernel.shape[-2]
        c1 = int(conv_kernel_pad.shape[4]//2 - conv_kernel.shape[-1]//2)
        c2 = c1 + conv_kernel.shape[-1]
        conv_kernel_pad[..., r1:r2, c1:c2] = conv_kernel
        W = ifft2d(conv_kernel_pad, dims=(-1,-2)) * conv_kernel_pad.shape[3] * conv_kernel_pad.shape[4] # torch.Size([2, 16, 16, 256, 256])

        # # confirms that image-space unaliasing using grappa weights works 
        # img_acc = ifft2d(data, dims=(-1,-2))[:,None,...]
        # rec_img = torch.sum(img_acc * W, dim=2)
        # rss_img = torch.sqrt(torch.sum(torch.abs(rec_img * rec_img.conj()), dim=1))
        # plt.figure()
        # plt.imshow(rss_img[0,:,:].cpu(), cmap='gray')
        # plt.show()
        # plt.figure()
        # plt.imshow(rss_img[1,:,:].cpu(), cmap='gray')
        # plt.show()

        # coil combination weights
        rec = ifft2d(out, dims=(-1,-2)) 
        rss = torch.sqrt(torch.sum(torch.abs(rec * rec.conj()), dim=1, keepdim=True))
        p = rec.conj() / rss # [2, 16, 256, 256]
        p = p.permute(0,2,3,1)[...,None]
        pT = p.permute(0,1,2,4,3)

        # calculate g-factor
        W = W.permute(0,3,4,1,2)
        WH = torch.conj(W.permute(0,1,2,4,3))
        if noise_cov is not None:
            psi = noise_cov[None,None,None,:,:].to(torch.complex64)
        else: 
            psi = torch.eye(W.shape[-1], dtype=W.dtype, device=data.device)[None,None,None,:,:]
        I = torch.eye(W.shape[-1], dtype=W.dtype, device=data.device)[None,None,None,:,:]
        pTI = pT @ I 
        pTIH = torch.conj(pTI.permute(0,1,2,4,3))
        pTW = pT @ W 
        pTWH = torch.conj(pTW.permute(0,1,2,4,3))
        gfactor = torch.sqrt( torch.abs(torch.linalg.diagonal( pTW @ psi @ pTWH )) ) / torch.sqrt(torch.abs(torch.linalg.diagonal(pTI @ psi @ pTIH)))
        gfactor = gfactor[...,0] 

        return slc_ksp, rss, gfactor
