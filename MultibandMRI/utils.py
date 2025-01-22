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

def get_kernel_patches(
        inp: Tensor,
        kernel_size: Tuple=(5,5),
        accel: Tuple=(1,1),
        stride: Tuple=None,
        pad=False
):
    
    if pad:
        eff_row_kernel_size = (kernel_size[0] - 1) * accel[0] + 1
        eff_col_kernel_size = (kernel_size[1] - 1) * accel[1] + 1
        row_padding = (eff_row_kernel_size - 1)//2 
        col_padding = (eff_col_kernel_size - 1)//2
        inp = torch.nn.functional.pad(inp, (col_padding, col_padding, row_padding, row_padding), mode='constant', value=0)

    stride = stride if stride is not None else (1,1)
    patches = torch.nn.functional.unfold(inp, kernel_size=kernel_size, dilation=accel, stride=stride)
    patches = patches.transpose(1,2)
    patches = patches[:,None,:,:]
    return patches 
    
def get_kernel_points(
        inp: Tensor,
        shifts: Tuple,
        kernel_size: Tuple,
        accel: Tuple=(1,1),
):
    rows, cols = inp.shape[2], inp.shape[3]
    eff_row_kernel_size = (kernel_size[0] - 1) * accel[0] + 1
    eff_col_kernel_size = (kernel_size[1] - 1) * accel[1] + 1
    nr = rows - eff_row_kernel_size + 1
    nc = cols - eff_col_kernel_size + 1
    inp_shifted = torch.roll(inp, shifts=(-shifts[0], -shifts[1]), dims=(2,3))
    inp_shifted = inp_shifted[:,:,:nr,:nc]
    points = torch.nn.functional.unfold(inp_shifted, kernel_size=(1,1), stride=(1,1))
    points = points[...,None]
    return points 

def get_kernel_shifts(kernel_size: Tuple,
                      accel: Tuple
):
    eff_row_kernel_size = (kernel_size[0] - 1) * accel[0] + 1
    eff_col_kernel_size = (kernel_size[1] - 1) * accel[1] + 1
    eff_kernel_size = (eff_row_kernel_size, eff_col_kernel_size)
    base_read_shift = eff_row_kernel_size // 2 
    base_phase_shift = eff_col_kernel_size // 2
    shifts = []
    start_inds = []
    for rfe in range(accel[0]):
        for rpe in range(accel[1]):
            shifts.append((base_read_shift + rfe, base_phase_shift + rpe))
            start_inds.append((rfe, rpe))
    return shifts, start_inds, eff_kernel_size

def interp_to_matrix_size(inp: Tensor,
                          matrix_size:Tuple
):

    rowpad = matrix_size[0] - inp.shape[2]
    rowpre = rowpad//2
    rowpst = rowpad - rowpre 
    colpad = matrix_size[1] - inp.shape[3]
    colpre = colpad//2
    colpst = colpad - colpre 
    return torch.nn.functional.pad(inp, (colpre, colpst, rowpre, rowpst), mode='constant', value=0)
        
def get_num_interpolated_points(shp: Tuple,
                                kernel_size: Tuple,
                                accel: Tuple
):
    
    rows, cols = shp[2], shp[3]
    eff_row_kernel_size = (kernel_size[0] - 1) * accel[0] + 1
    eff_col_kernel_size = (kernel_size[1] - 1) * accel[1] + 1
    nr = rows - eff_row_kernel_size + 1
    nc = cols - eff_col_kernel_size + 1
    rv = torch.arange(0, nr, accel[0])
    cv = torch.arange(0, nc, accel[1])
    nr = torch.numel(rv)
    nc = torch.numel(cv)
    return nr, nc 