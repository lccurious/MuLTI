import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import NLayerLeakyMLP


class GroupLinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks, diagonal, hidden, params=None):
        super(GroupLinearLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.diagonal = diagonal
        self.hidden = hidden

        if diagonal:
            # for independent components transitions
            self.d = nn.Parameter(0.01 * torch.randn(num_blocks, out_dim))
        else:
            if hidden is None:
                # for full rank causal transitions
                self.w = nn.Parameter(0.01 * torch.randn(num_blocks, in_dim, out_dim))
            else:
                # for low rank causal transitions
                self.wh = nn.Parameter(0.01 * torch.randn(num_blocks, in_dim, hidden))
                self.hw = nn.Parameter(0.01 * torch.randn(num_blocks, hidden, out_dim))

        if params is not None:
            self.w = nn.Parameter(params)
    
    def forward(self, x: torch.Tensor):

        if self.diagonal:
            w = torch.diag_embed(self.d)
            # x: [BS, num_blocks, in_dim] -> [num_blocks, BS, in_dim]
            x = x.permute(1, 0, 2)
            x = torch.bmm(x, w)
            # x: [BS, num_blocks, out_dim]
            x = x.permute(1, 0, 2)
        elif self.hidden is None:
            x = x.permute(1, 0, 2)
            x = torch.bmm(x, self.w)
            x = x.permute(1, 0, 2)
        else:
            x = x.permute(1, 0, 2)
            x = torch.bmm(x, self.wh)
            x = torch.bmm(x, self.hw)
            x = x.permute(1, 0, 2)
        return x
    
    def get_weight_matrix(self):
        if self.diagonal:
            return torch.diag_embed(self.d)
        elif self.hidden is None:
            return self.w
        else:
            return torch.matmul(self.wh, self.hw)


class VARTransitionPrior(nn.Module):
    def __init__(self, lags, latent_size, bias=False, params=None):
        super().__init__()
        # self.init_hiddens = nn.Parameter(0.001 * torch.randn(lags, latent_size))    
        # out[:,:,0] = (x[:,:,0]@conv.weight[:,:,0].T)+(x[:,:,1]@conv.weight[:,:,1].T) 
        # out[:,:,1] = (x[:,:,1]@conv.weight[:,:,0].T)+(x[:,:,2]@conv.weight[:,:,1].T)
        self.L = lags      
        self.transition = GroupLinearLayer(in_dim=latent_size, 
                                           out_dim=latent_size, 
                                           num_blocks=lags,
                                           diagonal=False,
                                           hidden=None,
                                           params=params)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(0.001 * torch.randn(1, latent_size))
    
    def forward(self, x, mask=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)

        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:, -1:], x[:, :-1]
        # residuals = torch.sum(self.transition(yy), dim=1) - xx.squeeze()
        # residuals = residuals.reshape(batch_size, -1, input_dim)
        x_hat = torch.sum(self.transition(yy), dim=1).reshape(batch_size, -1, input_dim)
        return x_hat
