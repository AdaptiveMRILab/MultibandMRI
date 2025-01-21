import torch 
from torch import Tensor 
from typing import Tuple 

def fft1d(inp, dim):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(inp, dim=dim), dim=dim), dim=dim)
    
def ifft1d(inp, dim):
    return torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(inp, dim=dim), dim=dim), dim=dim)
    
def fft2d(inp, dims=(0,1)):
    return fft1d(fft1d(inp, dims[0]), dims[1])

def ifft2d(inp, dims=(0,1)):
    return ifft1d(ifft1d(inp, dims[0]), dims[1])

def get_kspace_patches(
        inp: Tensor, # [batch_size, coils, rows, cols]
        kernel_size=(5,5),
        accel=(1,1),
        pad=False
):
    
    coils = inp.shape[1]
    
    # get the stride=1 total kernel sizes 
    kern_rows = kernel_size[0]*accel[0] - 1
    kern_cols = kernel_size[1]*accel[1] - 1
    ksize = (kern_rows, kern_cols)

    if pad:
        K,L = ksize 
        pad_height_total = K - 1
        pad_width_total = L - 1
        pad_top = pad_height_total // 2
        pad_bottom = pad_height_total - pad_top
        pad_left = pad_width_total // 2
        pad_right = pad_width_total - pad_left
        inp = torch.nn.functional.pad(inp, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    # source point mask 
    mask = torch.zeros((1, coils, kern_rows, kern_cols), dtype=torch.float32, device=inp.device)
    mask[:,:,::accel[0],::accel[1]] = 1.0
    mask = torch.nn.functional.unfold(mask, kernel_size=ksize)
    mask = mask[0,:,0]
    indices = torch.nonzero(mask).squeeze()

    # get multi-coil patches of k-space from source points 
    # these patches will have size [batch_size, num_patches, coils*kernel_size[0]*kernel_size[1]]
    patches = torch.nn.functional.unfold(inp, kernel_size=ksize)
    patches = patches[:,indices,:]

    # return the patches with a singleton channel dimension 
    return patches[:,None,:,:]

def get_kspace_points(
        inp: Tensor,
        shifts: Tuple,
        kernel_size=(5,5),
        accel=(1,1),
):
    
    coils = inp.shape[1]
    
    # get the stride=1 total kernel sizes 
    kern_rows = kernel_size[0]*accel[0] - 1
    kern_cols = kernel_size[1]*accel[1] - 1
    ksize = (kern_rows, kern_cols)

    # source point mask 
    mask = torch.zeros((1, coils, kern_rows, kern_cols), dtype=torch.float32, device=inp.device)
    mask[:,:,shifts[0],shifts[1]] = 1.0
    mask = torch.nn.functional.unfold(mask, kernel_size=ksize)
    mask = mask[0,:,0]
    indices = torch.nonzero(mask).squeeze()

    # get point within multi-coil patches of k-space
    patches = torch.nn.functional.unfold(inp, kernel_size=ksize)
    patches = patches[:,indices,:] 

    # return the patches with a singleton 4th dimension
    # it will have a shape [batch_size, coils, num_kernels, 1]
    return patches[...,None]




    
   

def extract_and_flatten_multicoil_kspace_patches(
        inp: Tensor,       # k-space tensor of shape [batch_size, coils, nfe, np]
        kernel_size=(5,5), # tuple of (frequency, phase) encoding kernel sizes 
        stride=(1,1),       # acceleration factor along each dimension 
        pad=False,
):
    
    if pad:
        K,L = kernel_size 
        pad_height_total = K - 1
        pad_width_total = L - 1
        pad_top = pad_height_total // 2
        pad_bottom = pad_height_total - pad_top
        pad_left = pad_width_total // 2
        pad_right = pad_width_total - pad_left
        inp = torch.nn.functional.pad(inp, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    batch_size = inp.shape[0]
    patches = torch.nn.functional.unfold(inp, kernel_size=kernel_size, stride=stride)
    num_patches = patches.shape[2]
    patches = patches.transpose(1,2).reshape(batch_size, num_patches, -1)

    return patches 

def extract_point_within_multicoil_kspace_patch(
        inp: Tensor,
        kernel_size=(5,5),
        stride=(1,1),
        shift=(2,2)  
):
    batch_size, num_coils, _, _ = inp.shape
    patches = torch.nn.functional.unfold(inp, kernel_size=kernel_size, stride=stride)
    num_patches = patches.shape[2]
    row_shift, col_shift = shift
    row_kernel_size, col_kernel_size = kernel_size
    shifted_index = row_shift * col_kernel_size + col_shift
    extracted_point = patches.reshape(batch_size, num_coils, -1, num_patches)[:, :, shifted_index, :]
    return extracted_point