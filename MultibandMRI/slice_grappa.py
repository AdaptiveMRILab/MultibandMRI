import torch 
from torch import Tensor
from typing import Tuple 
from MultibandMRI import get_kernel_patches, get_kernel_points, get_kernel_shifts, get_num_interpolated_points, interp_to_matrix_size, ifft2d
from MultibandMRI.utils import compute_weight_variances, prepare_sampling_density, compute_rss_variance_from_weights

class slice_grappa:

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
            b = get_kernel_points(calib_data, shifts=shifts, kernel_size=self.kernel_size, accel=self.accel)
            self.weights.append(AHA_inv @ (AH @ b))

    def apply(self,
              data,
              noise_cov: Tensor=None,
              return_gfactor: bool=False):
        '''
        Args:
            data: undersampled multiband k-space (sms, coils, readout, phase)
            noise_cov: optional (coils x coils) noise covariance; defaults to identity
            sampling_density: optional 1D/phase vector for variable-density undersampling
            return_gfactor: if True, also return analytical g-factor map (rss domain)
        '''

        # figure out number of interpolated points along each dimension 
        nr, nc = get_num_interpolated_points(data.shape, self.kernel_size, self.accel)

        # interpolate the missing points
        A = get_kernel_patches(data, kernel_size=self.kernel_size, accel=self.accel, stride=self.accel)
        Y = [(A@w).view(self.sms, self.coils, nr, -1) for w in self.weights]
        out = torch.zeros((self.sms, self.coils, self.accel[0]*nr, self.accel[1]*nc), dtype=data.dtype, device=data.device)
        for rfe, rpe in self.start_inds:
            out[:,:,rfe::self.accel[0],rpe::self.accel[1]] = Y[rfe*self.accel[1]+rpe]

        # zero-fill to final matrix size 
        if self.final_matrix_size is not None:
            out = interp_to_matrix_size(out, self.final_matrix_size)

        # get coil-combined image 
        img = ifft2d(out, dims=(2,3))
        rss = torch.sqrt(torch.sum(torch.abs(img * img.conj()), dim=1))

        if not return_gfactor:
            return out, rss
        
        # get composite convolution kernel 
        conv_kernels = torch.stack([ torch.reshape(w, (self.sms, self.coils, self.coils) + self.kernel_size) for w in self.weights], dim=-1)
        conv_kernel = torch.zeros((self.sms, self.coils, self.coils, self.accel[0]*self.kernel_size[0], self.accel[1]*self.kernel_size[1]), dtype=conv_kernels.dtype, device=data.device)
        for rfe in range(self.accel[0]):
            for rpe in range(self.accel[1]):
                conv_kernel[..., rfe::self.accel[0], rpe::self.accel[1]] = torch.flip(conv_kernels[..., rfe*self.accel[1]+rpe], dims=(-1,-2))
        
        # zero-pad the kernel and bring it to the image domain 
        conv_kernel_pad = torch.zeros((self.sms, self.coils, self.coils) + self.final_matrix_size, dtype=conv_kernel.dtype, device=data.device)
        r1 = int(self.final_matrix_size[0]//2 - conv_kernel.shape[-2]//2) 
        r2 = r1 + conv_kernel.shape[-2]
        c1 = int(self.final_matrix_size[1]//2 - conv_kernel.shape[-1]//2)
        c2 = c1 + conv_kernel.shape[-1]
        conv_kernel_pad[..., r1:r2, c1:c2] = conv_kernel
        W = ifft2d(conv_kernel_pad, dims=(-1,-2)) * self.final_matrix_size[0] * self.final_matrix_size[1] # torch.Size([2, 16, 16, 256, 256])

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

        

        return out, rss, gfactor
