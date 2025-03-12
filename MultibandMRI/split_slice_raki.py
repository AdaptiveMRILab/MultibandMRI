import torch 
from torch import Tensor
from typing import Tuple 
import os 
from MultibandMRI import get_kernel_patches, get_kernel_points, get_kernel_shifts, get_num_interpolated_points, interp_to_matrix_size, ifft2d, train_complex_net, load_complex_net

class split_slice_raki:

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

    def calibrate(self, calib_data):
    
        # split slice grappa does not require explicit sum of calibration data over slices
        A = get_kernel_patches(calib_data, kernel_size=self.kernel_size, accel=self.accel)
        A = torch.cat([A[None,s,:,:,:] for s in range(self.sms)], dim=2)
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
        I = torch.eye(self.sms, dtype=torch.float32, device=calib_data.device)
        for shifts in self.kernel_shifts:

            y = get_kernel_points(calib_data, shifts=shifts, kernel_size=self.kernel_size, accel=self.accel)
            b = torch.stack([torch.cat([y[d,...] * I[d,n] for n in range(self.sms)],1) for d in range(self.sms)],dim=0)
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

    def apply(self, data):

        # figure out number of interpolated points along each dimension 
        nr, nc = get_num_interpolated_points(data.shape, self.kernel_size, self.accel)

        # get the source data kernel patches 
        A = get_kernel_patches(data, kernel_size=self.kernel_size, accel=self.accel, stride=self.accel)

        # input to neural network 
        X = A[0,0,:,:]
        if self.scale_data:
            xmean = torch.mean(X, dim=1, keepdim=True)
            xstd = torch.std(X, dim=1, keepdim=True)
            X = (X - xmean) / xstd

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
                model = load_complex_net(self.model_paths[k][s], self.net_type, X.shape[1], self.coils, num_layers=self.num_layers, hidden_size=self.hidden_size).to(X.device)
                pred = model(X)
                if self.scale_data:
                    pred = pred*xstd + xmean 
                pred = pred.permute(1,0).view(self.coils, nr, -1)
                out[s,:,rfe::self.accel[0],rpe::self.accel[1]] = self.linear_weight * out_linear[s,:,rfe::self.accel[0],rpe::self.accel[1]] + pred 
                    
        # zero-fill to final matrix size 
        if self.final_matrix_size is not None:
            out = interp_to_matrix_size(out, self.final_matrix_size)

        # get coil-combined image 
        img = ifft2d(out, dims=(2,3))
        rss = torch.sqrt(torch.sum(torch.abs(img * img.conj()), dim=1))

        return out.detach(), rss.detach()