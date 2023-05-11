from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def gen_assignment(cost_matrix):
    row, col = linear_sum_assignment(cost_matrix)
    np_assignment_matrix = coo_matrix(
        (np.ones_like(row), (row, col))).toarray()
    return np_assignment_matrix


def gumbel_matching(log_alpha: torch.Tensor, noise: bool = True) -> torch.Tensor:
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise)
    np_log_alpha = log_alpha.detach().to("cpu").numpy()
    np_assignment_matrices = gen_assignment(np_log_alpha)
    np_assignment_matrices = np.stack(np_assignment_matrices, 0)
    assignment_matrices = torch.from_numpy(
        np_assignment_matrices).float().to(log_alpha.device)
    return assignment_matrices


def gumbel_matching_indices(log_alpha: torch.Tensor, share_dim: int, noise: bool = True) -> torch.Tensor:
    assingment_matrix = gumbel_matching(log_alpha, noise=noise)
    share_indices = assingment_matrix.argmax(0)[:share_dim]
    dim = log_alpha.size(0)
    t1 = torch.arange(dim, device=log_alpha.device)
    t2 = share_indices
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    perm_indices = torch.cat((intersection, difference)).long().to(log_alpha.device)
    return perm_indices


def conv_init(m):
    """Initialize the convolution kernel

    :param m: Conv module
    :type m: nn.Module
    """
    nn.init.kaiming_normal_(m.weight, mode='fan_out')
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)


def bn_init(bn, scale):
    """Initialize the batch norm kernel

    :param bn: Batchnorm module
    :type bn: nn.Module
    :param scale: The normalize scale
    :type scale: float
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
