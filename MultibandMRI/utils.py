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
    rowpst = rowpad//2
    rowpre = rowpad - rowpst 
    colpad = matrix_size[1] - inp.shape[3]
    colpst = colpad//2
    colpre = colpad - colpst
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

class complex_relu(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(complex_relu, self).__init__()
        self.eps = eps
    def forward(self, x):
        mag = torch.abs(x)
        return torch.nn.functional.relu(mag).to(torch.complex64)/(mag+self.eps)*x

class complex_mlp(torch.nn.Module):
    '''
    A complex-valued multi-layer perceptron
    '''
    def __init__(self, in_size, out_size, num_layers=4, hidden_size=64, bias=False):
        super(complex_mlp, self).__init__()
        self.num_layers = num_layers
        self.layers_real = torch.nn.ModuleList()
        self.layers_imag = torch.nn.ModuleList()
        for n in range(num_layers):
            ninp = in_size if n == 0 else hidden_size
            nout = out_size if n == num_layers - 1 else hidden_size 
            self.layers_real.append(torch.nn.Linear(in_features=ninp, out_features=nout, bias=bias))
            self.layers_imag.append(torch.nn.Linear(in_features=ninp, out_features=nout, bias=bias))
        self.crelu = complex_relu()

    def forward(self, x):
        for n in range(self.num_layers):
            xr = self.layers_real[n](x.real) - self.layers_imag[n](x.imag)
            xi = self.layers_real[n](x.imag) + self.layers_imag[n](x.real)
            x = torch.complex(xr, xi) 
            if n < self.num_layers - 1:
                x = self.crelu(x)
        return x 
    
def train_complex_mlp(X, Y, model_path, train_split=0.75, num_layers=4, hidden_size=128, bias=False, num_epochs=100, learn_rate=1e-4, scale_data=True, random_seed=42):

    torch.manual_seed(random_seed) 

    # get the training and validation indices
    nsamples = X.shape[0]
    shuffled_inds = torch.randperm(nsamples)
    ntrain = int(nsamples * train_split)
    train_inds = shuffled_inds[:ntrain]
    val_inds = shuffled_inds[ntrain:]

    # make the model 
    in_size = X.shape[1]
    out_size = Y.shape[1]
    model = complex_mlp(in_size, out_size, num_layers=num_layers, hidden_size=hidden_size, bias=bias).to(X.device) 

    # set up the optimizer and loss functions 
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = torch.nn.MSELoss()

    train_loss = torch.zeros((num_epochs,), dtype=torch.float32, device=X.device)
    val_loss = torch.zeros((num_epochs,), dtype=torch.float32, device=X.device)
    best_val_loss = 1e9 

    if scale_data:
        xmean = torch.mean(X, dim=1, keepdim=True)
        xstd = torch.std(X, dim=1, keepdim=True)
        X = (X - xmean) / xstd
        Y = (Y - xmean) / xstd  
    
    for epoch in range(num_epochs):
        
        # training step
        model.train()
        optimizer.zero_grad()
        pred_train = model(X[train_inds,:])
        loss = loss_fn(pred_train.real, Y[train_inds,:].real) + \
               loss_fn(pred_train.imag, Y[train_inds,:].imag)
        train_loss[epoch] = loss.item()
        loss.backward()
        optimizer.step()

        # validation 
        model.eval()
        pred_val = model(X[val_inds,:])
        loss = loss_fn(pred_val.real, Y[val_inds,:].real) + \
               loss_fn(pred_val.imag, Y[val_inds,:].imag)
        val_loss[epoch] = loss.item()

        # save the current model if it improves validation performance 
        if val_loss[epoch] < best_val_loss:
            best_val_loss = val_loss[epoch]
            torch.save(model.state_dict(), model_path)
    
    # load the final model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model, train_loss, val_loss

def load_complex_mlp(model_path, in_size, out_size, num_layers, hidden_size):
    model = complex_mlp(in_size, out_size, num_layers=num_layers, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model 
