import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import geoopt

from multi.core import losses
from multi.core import metrics
from multi.core.functions import gumbel_matching_indices
from multi.models.transitions import VARTransitionPrior
from multi.models.mlp import Discriminator
from multi.models import encoders
from multi.models import invertible_network_utils
from typing import Any
import logging

logging.getLogger("ligthning.pytorch").setLevel(logging.INFO)

# configure logging on module level, redirect to file
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.StreamHandler())


def is_close_to_permutation(mat, threshold=1e-3):
    # Ensure all values are between 0 and 1
    if torch.any((mat < 0) | (mat > 1)):
        return False
    
    rows, cols = mat.size()
    
    # Vector of ones for comparison
    ones_vector = torch.ones(rows, dtype=torch.float32, device=mat.device)
    
    # Check if the sum of rows and columns are close to 1
    if not torch.allclose(mat.sum(dim=0), ones_vector) or not torch.allclose(mat.sum(dim=1), ones_vector):
        return False

    # Threshold the matrix to get close-to-binary values
    binary_mat = (mat > (1 - threshold)).float()

    # Ensure it's a valid permutation matrix
    if not torch.equal(binary_mat.sum(dim=0), ones_vector) or not torch.equal(binary_mat.sum(dim=1), ones_vector):
        return False

    return True


class MuLTI(pl.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.lr = cfg.lr
        self.time_lag = cfg.time_lag
        self.n = cfg.n
        self.n_views = cfg.num_views
        self.n_shared = cfg.n_shared
        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        self.beta3 = cfg.beta3

        assert (
            cfg.n - cfg.n_shared) % self.n_views == 0, "Private dims can not be divided!"
        self.n_private = (cfg.n - cfg.n_shared) // self.n_views

        if cfg.p:
            self.loss = losses.LpSimCLRLoss(
                p=cfg.p, tau=cfg.tau, simclr_compatibility_mode=True
            )
        else:
            self.loss = losses.SimCLRLoss(normalize=False, tau=cfg.tau)

        self.f_transition = VARTransitionPrior(
            self.cfg.time_lag, self.cfg.n, self.cfg.n)

        output_normalization = None
        self.g_list = nn.ModuleList()
        self.h_ind_list = []
        self.f_list = nn.ModuleList()
        self.h_list = []
        self.brikhoff_polytopes = nn.ParameterList()

        self.learn_permutation = False
        self.perm_list = nn.ModuleList()

        if self.n_private != 0:
            self.view_dims = [torch.cat([
                torch.arange(cfg.n_shared),
                torch.arange(cfg.n_shared + i, cfg.n_shared +
                             i + self.n_private)
            ]) for i in torch.arange(0, cfg.n - cfg.n_shared, step=self.n_private)]
        else:
            self.view_dims = [torch.arange(cfg.n) for _ in range(self.n_views)]
        for i in range(self.n_views):
            n_dim = cfg.n_shared + self.n_private
            g = invertible_network_utils.construct_invertible_mlp(
                n=n_dim,
                n_layers=cfg.n_mixing_layer,
                act_fct=cfg.act_fct,
                cond_thresh_ratio=0.0,
                n_iter_cond_thresh=25000,
            )
            for p in g.parameters():
                p.requires_grad = False
            self.g_list.append(g)
            self.h_ind_list.append(lambda z: self.g_list[i](z))
            self.f_list.append(encoders.get_mlp(
                n_in=n_dim,
                n_out=n_dim,
                layers=[
                    n_dim * 10,
                    n_dim * 50,
                    n_dim * 50,
                    n_dim * 50,
                    n_dim * 50,
                    n_dim * 10
                ],
                output_normalization=output_normalization,
            ))
            self.h_list.append(lambda z: self.f_list[i](self.g_list[i](z)))
            brikhoff_manifold = geoopt.BirkhoffPolytope(tol=1e-5)
            with torch.no_grad():
                self.brikhoff_polytopes.append(
                    geoopt.ManifoldParameter(
                        # TODO: initialize with the Doubly Stochastic Matrices
                        torch.ones(n_dim, n_dim) / n_dim, manifold=brikhoff_manifold).proj_()
                )

        self.discriminator = Discriminator(cfg.n * (cfg.length - cfg.time_lag))
        self.save_hyperparameters()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    @staticmethod
    def permute_dims(z):
        assert z.dim() == 2

        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B, device=z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)

        return torch.cat(perm_z, 1)

    def on_train_start(self):
        self.logger.log_hyperparams(self.cfg, {
            "val/perm.disentanglement": 0,
            "val/lin.disentanglement": 0
        })

    def forward(self, x):
        _x_list = []
        for i, h in enumerate(self.h_list):
            _x_list.append(h(x))
        return _x_list

    def training_step(self, batch, batch_idx):
        z1, ct = batch
        batch_size = z1.size(0)
        _z1_shared_list = []
        _z1_private_list = []
        f_opt, d_opt, b_opt = self.optimizers()
        for i, h in enumerate(self.h_list):
            _z1_rec = h(z1[:, :, self.view_dims[i]])

            if self.learn_permutation:
                _z1_rec = torch.einsum(
                    "ijk,kk->ijk", _z1_rec.detach(), self.brikhoff_polytopes[i])
            else:
                perm_indices = gumbel_matching_indices(
                    self.brikhoff_polytopes[i], self.n_shared, False)
                _z1_rec = _z1_rec[..., perm_indices]

            if self.n_shared != 0:
                _z1_shared_list.append(_z1_rec[..., :self.n_shared])
            if self.n_private != 0:
                _z1_private_list.append(_z1_rec[..., self.n_shared:])

        if self.n_shared != 0:
            _z1_rec_shared_var, _z1_rec_shared = torch.var_mean(
                torch.stack(_z1_shared_list), dim=0)
            _z1_rec = torch.cat([
                _z1_rec_shared
            ] + _z1_private_list, dim=-1)
        else:
            _z1_rec = torch.cat(_z1_private_list, dim=-1)
            _z1_rec_shared_var = torch.zeros(1)

        if self.learn_permutation:
            # check if the doubly stochastic matrices are close to the standard permutation matrix
            for mat_id, ds_mat in enumerate(self.brikhoff_polytopes):
                self.learn_permutation = True
                if is_close_to_permutation(ds_mat):
                    ds_mat.requires_grad = False
                    self.learn_permutation = False
                    print(f"The {mat_id}-th doubly stochastic reach the permutation formula")
            
            b_opt.zero_grad()
            loss = _z1_rec_shared_var.mean()
            self.manual_backward(loss)
            b_opt.step()
            return loss

        z1_rec = _z1_rec[:, self.time_lag:]
        z2_con_z1_rec = self.f_transition(_z1_rec)
        z3_rec = torch.roll(z2_con_z1_rec, shifts=1, dims=0)
        self.log("train/multi-view.var", _z1_rec_shared_var.mean())

        residuals = z2_con_z1_rec - z1_rec
        if batch_idx % 2 == 0:
            D_z = self.discriminator(
                residuals.contiguous().view(batch_size, -1))
            tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
            loss_residuals = torch.norm(residuals, p=1, dim=-1).mean()
            total_loss_value, _, losses_value = self.loss(
                z1=None, z2_con_z1=None, z3=None,
                z1_rec=z1_rec.reshape(-1, self.n),
                z2_con_z1_rec=z2_con_z1_rec.reshape(-1, self.n),
                z3_rec=z3_rec.reshape(-1, self.n)
            )
            self.log("train/residuals", loss_residuals)

            loss = total_loss_value + self.beta1 * loss_residuals + \
                self.beta2 * _z1_rec_shared_var.mean() + self.beta3 * tc_loss
            self.log("train/loss", loss)
            f_opt.zero_grad()
            self.manual_backward(loss)
            f_opt.step()
            return loss
        else:
            residuals = residuals.detach()
            residuals, residuals_perm = torch.chunk(residuals, 2)
            D_z = self.discriminator(
                residuals.contiguous().view(batch_size // 2, -1))

            # Permute the other data batch
            ones = torch.ones(batch_size // 2, dtype=torch.long).to(z1.device)
            zeros = torch.zeros(
                batch_size // 2, dtype=torch.long).to(z1.device)

            residuals_perm = self.permute_dims(
                residuals_perm.contiguous().view(batch_size // 2, -1)).detach()
            D_z_pperm = self.discriminator(residuals_perm)
            D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) +
                               F.cross_entropy(D_z_pperm, ones))
            self.log("train/loss_tc", D_tc_loss)
            d_opt.zero_grad()
            self.manual_backward(D_tc_loss)
            d_opt.step()
            return D_tc_loss

    def validation_step(self, batch, batch_idx):
        z1, ct = batch
        _z1_shared_list = []
        _z1_private_list = []
        for i, h in enumerate(self.h_list):
            _z1_rec = h(z1[:, :, self.view_dims[i]])

            perm_indices = gumbel_matching_indices(
                self.brikhoff_polytopes[i], self.n_shared, False)
            _z1_rec = _z1_rec[..., perm_indices]

            if self.n_shared != 0:
                _z1_shared_list.append(_z1_rec[..., :self.n_shared])
            if self.n_private != 0:
                _z1_private_list.append(_z1_rec[..., self.n_shared:])

        if self.n_shared != 0:
            _z1_rec_var, _z1_rec_shared = torch.var_mean(
                torch.stack(_z1_shared_list), dim=0)
            z1_rec = torch.cat([
                _z1_rec_shared
            ] + _z1_private_list, dim=-1)
        else:
            z1_rec = torch.cat(_z1_private_list, dim=-1)
            _z1_rec_var = torch.zeros(1)
        z2_con_z1_rec = torch.cat(
            [z1_rec[:, :self.time_lag], self.f_transition(z1_rec)], dim=1)

        residuals = z2_con_z1_rec - z1_rec
        _z_disentanglement = z1.reshape(-1, self.n)
        _z_rec_disentanglement = z1_rec.reshape(-1, self.n)
        (linear_disentanglement_score, _), _ = metrics.linear_disentanglement(
            _z_disentanglement,
            _z_rec_disentanglement, mode="r2"
        )
        (permutation_disentanglement_score, _), _ = metrics.permutation_disentanglement(
            _z_disentanglement,
            _z_rec_disentanglement, mode="pearson",
            solver="munkres",
            rescaling=True,
        )
        self.logger.experiment.add_histogram(
            tag="val/residuals", values=torch.flatten(residuals[:, self.time_lag:]),
            global_step=self.current_epoch
        )
        self.log("val/perm.disentanglement", permutation_disentanglement_score)
        self.log("val/lin.disentanglement", linear_disentanglement_score)
        print(
            "val/perm.disentanglement: {:.5f}".format(permutation_disentanglement_score),
            "val/lin.disentanglement: {:.5f}".format(linear_disentanglement_score)
        )
        return {
            "val/perm.disentanglement": permutation_disentanglement_score,
            "val/lin.disentanglement":
            linear_disentanglement_score
        }
    
    def on_train_epoch_start(self) -> None:
        if self.current_epoch > -1 and self.current_epoch % 2 == 0:
            self.learn_permutation = True
        else:
            self.learn_permutation = False
    
        for p in self.brikhoff_polytopes:
            p.requires_grad = self.learn_permutation
        
        for f in self.f_list:
            for p in f:
                p.requires_grad = (not self.learn_permutation)
        
    def configure_optimizers(self):
        params = [
            {'params': self.f_transition.parameters()}
        ]
        for f in self.f_list:
            params.append({'params': f.parameters()})

        optimizer = torch.optim.AdamW(
            params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.001)
        optimizer_brickhoff = geoopt.optim.RiemannianAdam(
            self.brikhoff_polytopes, lr=self.lr)
        optimizer_discriminator = torch.optim.SGD(
            self.discriminator.parameters(), lr=self.lr / 2)
        return optimizer, optimizer_discriminator, optimizer_brickhoff
