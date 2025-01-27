import torch 
from torch import Tensor
from typing import Tuple 
import os 
from MultibandMRI import get_kernel_patches, get_kernel_points, get_kernel_shifts, get_num_interpolated_points, interp_to_matrix_size, ifft2d, train_complex_mlp, load_complex_mlp

class slice_raki:

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
                 learn_residual: bool=True):
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
        self.learn_residual = learn_residual
        self.num_layers = num_layers 
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.learn_rate = learn_rate
        self.random_seed = random_seed
        self.recon_folder = recon_folder
        self.train_split = train_split
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
        self.weights = [] # this will hold linear GRAPPA reconstruction weights 
        self.model_paths = []  # this will hold the trained RAKI model weights 
        for shifts in self.kernel_shifts:

            b = get_kernel_points(calib_data, shifts=shifts, kernel_size=self.kernel_size, accel=self.accel)
            w = AHA_inv @ (AH @ b)
            self.weights.append(w)

            # get the target data: it will either be the residual error after GRAPPA 
            # or simply the target k-space points 
            rhs = b - A@w if self.learn_residual else b 

            # train a model for each slice
            slice_model_paths = []
            for s in range(self.sms):
                model_path = os.path.join(self.recon_folder, 'model_shift%i_slice%i.pt'%(self.kernel_shifts.index(shifts), s))
                #print(A.shape)
                #X = torch.cat([A[0,n,:,:] for n in range(self.coils)], dim=1)   # [observations, prod(kernel_size)*coils*coils]
                X = A[0,0,:,:]
                Y = rhs[s,:,:,0].permute(1,0)                                   # [observations, coils]
                _, train_loss, val_loss  = train_complex_mlp(X, Y, model_path, self.train_split, 
                                          num_layers=self.num_layers, hidden_size=self.hidden_size, 
                                          num_epochs=self.num_epochs, learn_rate=self.learn_rate, 
                                          random_seed=self.random_seed)
                slice_model_paths.append(model_path)
            self.model_paths.append(slice_model_paths)

    def apply(self, data):

        # figure out number of interpolated points along each dimension 
        nr, nc = get_num_interpolated_points(data.shape, self.kernel_size, self.accel)

        # get the source data kernel patches 
        A = get_kernel_patches(data, kernel_size=self.kernel_size, accel=self.accel, stride=self.accel)

        # linear GRAPPA interpolation 
        Y = [(A@w).view(self.sms, self.coils, nr, -1) for w in self.weights]
        out_linear = torch.zeros((self.sms, self.coils, self.accel[0]*nr, self.accel[1]*nc), dtype=data.dtype, device=data.device)
        for rfe, rpe in self.start_inds:
            out_linear[:,:,rfe::self.accel[0],rpe::self.accel[1]] = Y[rfe*self.accel[1]+rpe]

        # do the nonlinear interpolation 
        out = torch.zeros((self.sms, self.coils, self.accel[0]*nr, self.accel[1]*nc), dtype=data.dtype, device=data.device)
        for k in range(len(self.start_inds)):
            rfe, rpe = self.start_inds[k]
            for s in range(self.sms):
                #X = torch.cat([A[0,n,:,:] for n in range(self.coils)], dim=1)
                X = A[0,0,:,:]
                model = load_complex_mlp(self.model_paths[k][s], X.shape[1], self.coils, num_layers=self.num_layers, hidden_size=self.hidden_size).to(X.device)
                pred = model(X).permute(1,0).view(self.coils, nr, -1)
                if self.learn_residual:
                    out[s,:,rfe::self.accel[0],rpe::self.accel[1]] = out_linear[s,:,rfe::self.accel[0],rpe::self.accel[1]] + pred 
                else:
                    out[s,:,rfe::self.accel[0],rpe::self.accel[1]] = pred
                    
        # zero-fill to final matrix size 
        if self.final_matrix_size is not None:
            out = interp_to_matrix_size(out, self.final_matrix_size)

        # get coil-combined image 
        img = ifft2d(out, dims=(2,3))
        rss = torch.sqrt(torch.sum(torch.abs(img * img.conj()), dim=1))

        return out, rss
    
