import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

from multi.core import spaces, latent_spaces


__all__ = ["LinearVAR"]


def l2_normalize(Amat, axis=0):
    l2norm = np.sqrt(np.sum(Amat * Amat, axis))
    Amat = Amat / l2norm
    return Amat


def generate_var_transtions(n, time_lags=2, cond_thresh_ratio=0.25, scale=1):
    cond_list = []
    for _ in range(100000):
        A = np.random.uniform(-1, 1, [n, n]) / scale
        A = l2_normalize(A, axis=0)
        cond_list.append(np.linalg.cond(A))
    cond_thresh = np.quantile(cond_list, cond_thresh_ratio)

    var_params = []
    for t in range(time_lags):
        condA = cond_thresh + 1
        while condA > cond_thresh:
            _param = np.random.uniform(-1, 1, [n, n]) / scale
            _param = l2_normalize(_param, axis=0)
            condA = np.linalg.cond(_param)
        print(
            f"VAR transition matrix {t+1}/{time_lags} generated with cond({condA:.2f}) < {cond_thresh:.2f}"
        )
        var_params.append(_param)
    var_params = np.array(var_params, dtype=np.float32)
    var_params = torch.from_numpy(var_params)
    return var_params


def linear_var(params, z_lags, length, noises):
    """Linear vector autoregression function.

    Args:
        params (torch.Tensor): [L, N, N] list of parameter matrices
        z_lags (torch.Tensor): [B, L, N] batch of multivariate time-series
        length (int): Number of time lags for autoregression

    Returns:
        torch.Tensor: The generated latents
    """
    batch_size = z_lags.size(0)
    num_lags = len(params)
    num_dim = z_lags.size(2)

    z_seq = torch.zeros((batch_size, length, num_dim), dtype=z_lags.dtype, device=z_lags.device)
    z_seq[:, :num_lags] = z_lags
    for t in range(num_lags, length):
        z_t = noises[:, t]
        for i, param in enumerate(params):
            z_t += z_seq[:, t - i - 1] @ param
        z_seq[:, t] = z_t
    return z_seq


def nonlinear_var(params, z_lags, length, noises):
    batch_size = z_lags.size(0)
    num_lags = len(params)
    num_dim = z_lags.size(2)

    z_seq = torch.zeros((batch_size, length, num_dim), dtype=z_lags.dtype, device=z_lags.device)
    z_seq[:, :num_lags] = z_lags
    for t in range(num_lags, length):
        z_t = noises[:, t]
        for i, param in enumerate(params):
            z_t += z_seq[:, t - i - 1] @ param
        z_seq[:, t] = F.leaky_relu(z_t, negative_slope=0.2)
    return z_seq


class LinearVARData(Dataset):
    def __init__(self, cfg, params, batch_size, num_samples) -> None:
        super().__init__()
        self.cfg = cfg
        self.n = cfg.n
        self.time_lag = cfg.time_lag
        self.batch_size = batch_size
        self.length = cfg.length
        self.num_samples = num_samples
        self.params = params
        self.device = 'cpu'

        self.space = spaces.NRealSpace(cfg.n) 

        self.eta = torch.zeros(cfg.n)
        if cfg.space_type == "sphere":
            self.eta[0] = 1.0
        
        if cfg.m_p:
            if cfg.m_p == 1:
                self.sample_marginal = lambda space, size, device=self.device: space.laplace(
                    self.eta, cfg.m_param, size, device
                )
            elif cfg.m_p == 2:
                self.sample_marginal = lambda space, size, device=self.device: space.normal(
                    self.eta, cfg.m_param, size, device
                )
            else:
                self.sample_marginal = lambda space, size, device=self.device: space.generalized_normal(
                    self.eta, cfg.m_param, p=cfg.m_p, size=size, device=device
                )
        else:
            self.sample_marginal = lambda space, size, device=self.device: space.uniform(size, device)
        
        if cfg.c_p:
            if cfg.c_p == 1:
                self.sample_conditional = lambda space, z, size, device=self.device: space.laplace(
                    z, cfg.c_param, size, device
                )
            elif cfg.c_p == 2:
                self.sample_conditional = lambda space, z, size, device=self.device: space.normal(
                    z, cfg.c_param, size, device
                )
            else:
                self.sample_conditional = lambda space, z, size, device=self.device: space.generalized_normal(
                    z, cfg.c_param, p=cfg.c_p, size=size, device=device
                )
        else:
            self.sample_conditional = lambda space, z, size, device=self.device: space.von_mises_fisher(
                z, cfg.c_param, size, device
            )
        
        self.latent_space = latent_spaces.LatentSpace(
            space=self.space,
            sample_marginal=self.sample_marginal,
            sample_conditional=self.sample_conditional,
        )

        self.noise_dists = []
        self.num_regimes = cfg.num_regimes
        self.regime_ids = torch.arange(cfg.num_regimes)
        noise_scales = torch.randint(0, 10000, (self.num_regimes,)) / 10000.0
        for ct in range(cfg.num_regimes):
            self.noise_dists.append(torch.distributions.Laplace(0.0, noise_scales[ct]))

    def __getitem__(self, index):
        z_lags = [self.latent_space.sample_marginal(self.batch_size) for _ in range(self.time_lag)]
        z_lags = torch.stack(z_lags, dim=1)
        z_lags = (z_lags - torch.mean(z_lags, dim=0, keepdim=True)) / torch.std(z_lags, axis=0, keepdim=True)

        # generate the noise for VAR
        num_samples_per_regime = self.batch_size // max(1, (self.num_regimes - 1))
        regime_ids = torch.arange(self.num_regimes).unsqueeze(1).repeat(1, num_samples_per_regime).flatten()
        _noises = torch.stack([
            self.latent_space.sample_conditional(
                torch.zeros_like(z_lags[:, 0]), self.batch_size) for t in range(self.length)
        ], dim=1)
        ct = torch.randperm(_noises.size(0))[:self.batch_size]

        z_seq = linear_var(self.params, z_lags, self.length, _noises[ct])
        return z_seq, regime_ids[ct]
    
    def __len__(self):
        return self.num_samples


class LinearVAR(pl.LightningDataModule):
    def __init__(self, cfg, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.cfg = cfg
        self.n = cfg.n
        self.num_samples_train = cfg.n_log_steps
        self.num_samples_val = 1
        self.time_lag = cfg.time_lag
        self.batch_size = cfg.batch_size
        self.var_params = generate_var_transtions(self.n, self.time_lag, scale=1)
    
    def transition_matrices(self):
        return torch.flip(self.var_params, dims=(0,))
    
    def setup(self, stage):
        self.var_train = LinearVARData(self.cfg, self.var_params, self.batch_size, self.num_samples_train)
        self.var_val = LinearVARData(self.cfg, self.var_params, 4096, self.num_samples_val)
    
    def train_dataloader(self):
        return DataLoader(self.var_train,
                          batch_size=None,  # to disable the automatic batching
                          num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.var_val,
                          batch_size=None,
                          num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.var_val,
                          batch_size=None,
                          num_workers=4)
