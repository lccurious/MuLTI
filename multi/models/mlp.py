import torch
import torch.nn as nn
from multi.core.functions import kaiming_init, normal_init

__all__ = ["NLayerLeakyMLP", "Discriminator"]


class NLayerLeakyMLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = 128
        self.net = nn.Sequential(
            nn.Linear(z_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_dim, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()
