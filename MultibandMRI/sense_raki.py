import torch 
from torch import Tensor
from typing import Tuple 
import os 
import numpy as np
from MultibandMRI import get_kernel_patches, get_kernel_points, get_kernel_shifts, get_num_interpolated_points, interp_to_matrix_size, ifft1d, ifft2d, fft1d, fft2d, train_complex_net, load_complex_net

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
        self.weights = [] # this will hold linear GRAPPA reconstruction weights 
        self.model_paths = []  # this will hold the trained RAKI model weights 
        for shifts in self.kernel_shifts:
            b = get_kernel_points(data, shifts=shifts, kernel_size=self.kernel_size, accel=self.accel)
            w = AHA_inv @ (AH @ b)
            self.weights.append(w)

            # get the target data (difference between acquired data and weighted GRAPPA-reconstructed data)
            rhs = b - self.linear_weight*A@w

            # train a model for each slice
            slice_model_paths = []
            for s in range(self.sms):
                model_path = os.path.join(self.recon_folder, 'model_shift%i_slice%i.pt'%(self.kernel_shifts.index(shifts), s))
                X = A[0,0,:,:]
                Y = rhs[s,:,:,0].permute(1,0)
                _, train_loss, val_loss  = train_complex_net(X, Y, model_path, self.net_type, self.train_split, 
                                            num_layers=self.num_layers, hidden_size=self.hidden_size, 
                                            num_epochs=self.num_epochs, learn_rate=self.learn_rate, 
                                            random_seed=self.random_seed, scale_data=self.scale_data,
                                            loss_function=self.loss_function, l2_frac=self.l2_frac)
                slice_model_paths.append(model_path)
            self.model_paths.append(slice_model_paths)


    def apply(self, inp_data):

        # readout FOV of extended-FOV images is no longer centered for an even number of simultaneously excited slices. add FOV/2 shift here
        if self.sms % 2 == 0: inp_data[:,:,1::2,:] = inp_data[:,:,1::2,:] * np.exp(1j*np.pi)

        # zero-fill data 
        data = torch.zeros((inp_data.shape[0], inp_data.shape[1], self.sms*inp_data.shape[2], inp_data.shape[3]), dtype=inp_data.dtype, device=inp_data.device)
        data[:,:,::self.sms,:] = inp_data 

        # figure out number of interpolated points along each dimension 
        nr, nc = get_num_interpolated_points(data.shape, self.kernel_size, self.accel)

        # interpolate the missing points
        A = get_kernel_patches(data, kernel_size=self.kernel_size, accel=self.accel, stride=self.accel)

        # input to neural network 
        X = A[0,0,:,:]
        if self.scale_data:
            xmean = torch.mean(X, dim=1, keepdim=True)
            xstd = torch.std(X, dim=1, keepdim=True)
            X = (X - xmean) / xstd

        # linear GRAPPA interpolation 
        Y = [(A@w).view(1, self.coils, nr, nc) for w in self.weights]
        out_linear = torch.zeros((1, self.coils, self.accel[0]*nr, self.accel[1]*nc), dtype=inp_data.dtype, device=inp_data.device)
        for rfe, rpe in self.start_inds:
            out_linear[:,:,rfe::self.accel[0],rpe::self.accel[1]] = Y[rfe*self.accel[1]+rpe]

        # do the nonlinear interpolation 
        out = torch.zeros((self.sms, self.coils, self.accel[0]*nr, self.accel[1]*nc), dtype=data.dtype, device=data.device)
        for k in range(len(self.start_inds)):
            rfe, rpe = self.start_inds[k]
            for s in range(self.sms):
                model = load_complex_net(self.model_paths[k][s], self.net_type, X.shape[1], self.coils, num_layers=self.num_layers, hidden_size=self.hidden_size).to(X.device)
                pred = model(X)
                if self.scale_data:
                    pred = pred*xstd + xmean 
                pred = pred.permute(1,0).view(self.coils, nr, -1)
                out[s,:,rfe::self.accel[0],rpe::self.accel[1]] = self.linear_weight * out_linear[s,:,rfe::self.accel[0],rpe::self.accel[1]] + pred 

        # final interpolation 
        if self.final_matrix_size is not None:
            adjusted_matrix_size = (self.sms*self.final_matrix_size[0], self.final_matrix_size[1])
            out = interp_to_matrix_size(out, adjusted_matrix_size)

        # data consistency
        out[torch.abs(data) > 0.0] = data[torch.abs(data) > 0.0]

        # bring to the image domain and crop slices
        nread = inp_data.shape[2]
        img = ifft2d(out, dims=(2,3))
        img = torch.stack([img[0,:,n*nread:(n+1)*nread,:] for n in range(self.sms)], axis=0)
        slc_ksp = fft2d(img, dims=(2,3))
        rss = torch.sqrt(torch.sum(torch.abs(img * img.conj()), dim=1))

        return slc_ksp, rss
