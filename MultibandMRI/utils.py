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
    out = torch.nn.functional.pad(inp, (colpre, colpst, rowpre, rowpst), mode='constant', value=0) # original 
    
    return out

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
    
class complex_resnet_block(torch.nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(complex_resnet_block, self).__init__()

        self.layer1_real = torch.nn.Linear(in_features=in_size, out_features=out_size, bias=bias)
        self.layer1_imag = torch.nn.Linear(in_features=in_size, out_features=out_size, bias=bias)

        self.layer2_real = torch.nn.Linear(in_features=out_size, out_features=out_size, bias=bias)
        self.layer2_imag = torch.nn.Linear(in_features=out_size, out_features=out_size, bias=bias)

        self.layer3_real = torch.nn.Linear(in_features=out_size, out_features=out_size, bias=bias)
        self.layer3_imag = torch.nn.Linear(in_features=out_size, out_features=out_size, bias=bias)

        self.crelu = complex_relu()

    def forward(self, x):

        x0 = x.clone()
        
        xr = self.layer1_real(x.real) - self.layer1_imag(x.imag)
        xi = self.layer1_real(x.imag) + self.layer1_imag(x.real)
        x = torch.complex(xr, xi) 
        x = self.crelu(x)

        xr = self.layer2_real(x.real) - self.layer2_imag(x.imag)
        xi = self.layer2_real(x.imag) + self.layer2_imag(x.real)
        x = torch.complex(xr, xi) 
        x = self.crelu(x)

        xr = self.layer3_real(x.real) - self.layer3_imag(x.imag)
        xi = self.layer3_real(x.imag) + self.layer3_imag(x.real)
        x = torch.complex(xr, xi) 

        x = self.crelu(x + x0)

        return x 
    
class complex_resnet(torch.nn.Module):
    def __init__(self, in_size, out_size, num_blocks=3, hidden_size=64, bias=False):
        super(complex_resnet, self).__init__()
        self.num_blocks = num_blocks

        self.crelu = complex_relu()
        
        self.linear1_real = torch.nn.Linear(in_features=in_size, out_features=hidden_size, bias=bias)
        self.linear1_imag = torch.nn.Linear(in_features=in_size, out_features=hidden_size, bias=bias)

        self.blocks = torch.nn.ModuleList()
        for n in range(self.num_blocks):
            self.blocks.append(complex_resnet_block(hidden_size, hidden_size, bias=bias))

        self.linear2_real = torch.nn.Linear(in_features=hidden_size, out_features=out_size, bias=bias)
        self.linear2_imag = torch.nn.Linear(in_features=hidden_size, out_features=out_size, bias=bias)


    def forward(self, x):

        xr = self.linear1_real(x.real) - self.linear1_imag(x.imag)
        xi = self.linear1_real(x.imag) + self.linear1_imag(x.real)
        x = torch.complex(xr, xi) 
        x = self.crelu(x)

        for n in range(self.num_blocks):
            x = self.blocks[n](x)

        xr = self.linear2_real(x.real) - self.linear2_imag(x.imag)
        xi = self.linear2_real(x.imag) + self.linear2_imag(x.real)
        x = torch.complex(xr, xi) 
        
        return x
    
class l1_l2_loss(torch.nn.Module):
    def __init__(self, l2_frac=0.5):
        super(l1_l2_loss, self).__init__()
        self.l2_frac = l2_frac
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
    def forward(self, input, target):
        if self.l2_frac == 0.0:
            return self.l1(input, target)
        elif self.l2_frac == 1.0:
            return self.l2(input, target)
        else:
            return (1.0 - self.l2_frac) * self.l1(input, target) + self.l2_frac * self.l2(input, target)

def train_complex_net(X, Y, model_path, net_type, train_split=0.75, num_layers=4, hidden_size=128, bias=False, num_epochs=100, learn_rate=1e-4, scale_data=True, random_seed=42, loss_function='L1', l2_frac=0.5):

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
    if net_type == 'MLP':
        model = complex_mlp(in_size, out_size, num_layers=num_layers, hidden_size=hidden_size, bias=bias).to(X.device) 
    elif net_type == 'RES':
        model = complex_resnet(in_size, out_size, num_blocks=num_layers, hidden_size=hidden_size, bias=bias).to(X.device) 
    elif net_type == 'MLPb':
        model = complex_mlp_bspline(in_size, out_size, num_layers=num_layers, hidden_size=hidden_size, bias=bias).to(X.device) 

    # set up the optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # set up the loss function 
    if loss_function == 'L2':
        loss_fn = torch.nn.MSELoss()
    elif loss_function == 'L1':
        loss_fn = torch.nn.L1Loss()
    elif loss_function == 'L1_L2':
        loss_fn = l1_l2_loss(l2_frac)

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

        # save the current model if it significantly improves validation performance 
        if 1.05*(val_loss[epoch]) < best_val_loss:
            best_val_loss = val_loss[epoch]
            torch.save(model.state_dict(), model_path)
    
    # load the final model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model, train_loss, val_loss

def load_complex_net(model_path, net_type, in_size, out_size, num_layers, hidden_size):
    # model = complex_mlp(in_size, out_size, num_layers=num_layers, hidden_size=hidden_size)
    if net_type == 'MLP':
        model = complex_mlp(in_size, out_size, num_layers=num_layers, hidden_size=hidden_size)
    elif net_type == 'RES':
        model = complex_resnet(in_size, out_size, num_blocks=num_layers, hidden_size=hidden_size)
    elif net_type == 'MLPb':
        model = complex_mlp_bspline(in_size, out_size, num_layers=num_layers, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model 

class CoilCompress:
    def __init__(self, data, vcoils, maxPoints=2000):
        super(CoilCompress,self).__init__()
        ncoils = data.shape[1]
        assert vcoils <= ncoils, 'Number of compressed virtual coils (%i) must be <= number of physical coils (%i)'%(vcoils,ncoils)
        self.ncoils = ncoils
        self.vcoils = vcoils
        self.maxPoints = maxPoints
        self.calcCompression(data)

    def calcCompression(self, data):
        dataMask = (torch.abs(data[:,0,...])>0.0).int()
        mtrx = torch.zeros((self.ncoils, torch.sum(dataMask)), dtype=data.dtype, device=data.device)
        for c in range(self.ncoils):
            mtrx[c,:] = data[:,c,...][dataMask > 0]
        if self.maxPoints is not None:
            inds = torch.argsort(-torch.sum(torch.abs(mtrx),dim=0))
            mtrx = mtrx[:,inds[:self.maxPoints]]
        u, _, _ = torch.linalg.svd(mtrx, full_matrices=False)
        self.U = u[:, :self.vcoils]
        self.Uh = torch.conj(self.U.T)
 
    def compress(self, data):
        dataMask = (torch.abs(data[:,0,...])>0.0).int()
        mtrx = torch.zeros((torch.sum(dataMask), self.ncoils), dtype=data.dtype, device=data.device)
        for c in range(self.ncoils):
            mtrx[:,c] = data[:,c,...][dataMask > 0]
        mtrx = mtrx @ self.U
        ccdata = torch.zeros((data.shape[0], self.vcoils, data.shape[2], data.shape[3]), dtype=data.dtype, device=data.device)
        for c in range(self.vcoils):
            tmp = torch.zeros_like(data[:,0,...])
            tmp[dataMask > 0] = mtrx[:, c]
            ccdata[:,c,...] = tmp.clone()
        return ccdata
    
# Code created by AI - try and get b-spline to work as a starting point

# # Linear activation function created by AI (works fine)
# class BSplineActivation(torch.nn.Module):
#     def __init__(self, num_ctrl_pts=8, degree=3):
#         super().__init__()
#         self.degree = degree
#         self.num_ctrl_pts = num_ctrl_pts
#         # Learnable control points
#         self.ctrl_pts = torch.nn.Parameter(torch.linspace(0, 1, num_ctrl_pts))
#         # Uniform knots
#         self.register_buffer('knots', torch.linspace(0, 1, num_ctrl_pts + degree + 1))

#     def forward(self, x):
#         # Piecewise linear interpolation as a simple B-spline approximation
#         idx = (x * (self.num_ctrl_pts - 1)).long()
#         idx = torch.clamp(idx, 0, self.num_ctrl_pts - 2)
#         left = self.ctrl_pts[idx]
#         right = self.ctrl_pts[idx + 1]
#         alpha = x * (self.num_ctrl_pts - 1) - idx.float()
#         return (1 - alpha) * left + alpha * right

def normalize_input(x, eps=1e-9):
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_norm = (x - x_min) / (x_max - x_min + eps)
    return x_norm, x_min, x_max, eps

class BSplineActivation(torch.nn.Module):
    # Default degree = 3
    # Default num_ctrl_pts = 8
    def __init__(self, num_ctrl_pts=8, degree=3):
        super().__init__()
        self.degree = degree
        self.num_ctrl_pts = num_ctrl_pts

        # Learnable control points
        self.ctrl_pts = torch.nn.Parameter(torch.linspace(0, 1, num_ctrl_pts))

        # Uniform knots (open uniform B-spline)
        knots = torch.concatenate([
            torch.zeros(degree),
            torch.linspace(0, 1, num_ctrl_pts - degree + 1),
            torch.ones(degree)
        ])
        print(len(knots))
        self.register_buffer('knots', knots)

    def forward(self, x):
        # x: (batch_size, layer_size)
        # returns: (batch_size, layer_size)

        # Normalize the input
        x_norm, x_min, x_max, eps = normalize_input(x)

        # Evaluate B-spline basis functions at x
        basis = self.bspline_basis(x_norm, self.degree, self.knots, self.num_ctrl_pts)

        # Weighted sum of control points
        output = torch.sum(basis * self.ctrl_pts, dim=-1)
        
        # Undo the normalization
        output = output * (x_max-x_min+eps) + x_min

        return output

    def bspline_basis(self, x, degree, knots, num_ctrl_pts):
        # x: (batch_size, layer_size)
        # knots: (batch_size, layer_size)
        # returns: (batch_size, layer_size, num_ctrl_pts)

        # Initialize zeroth degree basis functions
        basis = []
        for i in range(num_ctrl_pts):
            cond = (x >= knots[i]) & (x < knots[i+1])
            if i == num_ctrl_pts - 1:
                cond = (x >= knots[i]) & (x <= knots[i+1])
            basis.append(cond.float())
        basis = torch.stack(basis, dim=-1)

        # Loops through the degrees (each loop adds a degree)
        for d in range(1, degree+1):
            new_basis = []
            for i in range(num_ctrl_pts):
                left = torch.zeros_like(basis[..., i])
                left_den = knots[i+d] - knots[i]
                if left_den != 0:
                    left_num = x.squeeze(-1) - knots[i]
                    left = (left_num / left_den) * basis[..., i]
                right = torch.zeros_like(basis[..., i])
                if i+1 < num_ctrl_pts:
                    right_den = knots[i+d+1] - knots[i+1]
                    if right_den != 0:
                        right_num = knots[i+d+1] - x.squeeze(-1)
                        right = (right_num / right_den) * basis[..., i+1]
                new_basis.append(left + right)
            basis = torch.stack(new_basis, dim=-1)

        return basis
    
class complex_bspline(torch.nn.Module):
    # Default degree = 3
    # Default num_ctrl_pts = 8
    def __init__(self, eps=1e-6, num_ctrl_pts=8, degree=3):
        super(complex_bspline, self).__init__()
        self.eps = eps
        self.bspline = BSplineActivation(num_ctrl_pts=num_ctrl_pts, degree=degree)
    def forward(self, x):
        mag = torch.abs(x)
        activated = self.bspline(mag)
        return activated.to(torch.complex64) / (mag + self.eps) * x
    
class complex_mlp_bspline(torch.nn.Module):
    '''
    A complex-valued multi-layer perceptron with B-spline activation
    '''
    def __init__(self, in_size, out_size, num_layers=3, hidden_size=64, bias=False):
        super(complex_mlp_bspline, self).__init__()
        self.num_layers = num_layers
        self.layers_real = torch.nn.ModuleList()
        self.layers_imag = torch.nn.ModuleList()
        for n in range(num_layers):
            ninp = in_size if n == 0 else hidden_size
            nout = out_size if n == num_layers - 1 else hidden_size 
            self.layers_real.append(torch.nn.Linear(in_features=ninp, out_features=nout, bias=bias))
            self.layers_imag.append(torch.nn.Linear(in_features=ninp, out_features=nout, bias=bias))
        self.cbspline = complex_bspline()

    def forward(self, x):
        for n in range(self.num_layers):
            xr = self.layers_real[n](x.real) - self.layers_imag[n](x.imag)
            xi = self.layers_real[n](x.imag) + self.layers_imag[n](x.real)
            x = torch.complex(xr, xi) 
            if n < self.num_layers - 1:
                x = self.cbspline(x)
        return x
    