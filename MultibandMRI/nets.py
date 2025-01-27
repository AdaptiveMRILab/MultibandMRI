import torch 

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
            xr = self.layers_real(x.real) - self.layers_imag(x.imag)
            xi = self.layers_real(x.imag) + self.layers_imag(x.real)
            x = torch.complex(xr, xi) 
            if n < self.num_layers - 1:
                x = self.crelu(x)
        return x 
        
    