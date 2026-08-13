"""
sae.py — Sparse Autoencoder implementation
"""

# pylint: disable=invalid-name

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.sae.base import AutoEncoder


class SparseLinearAE(AutoEncoder):
    """Sparse Linear Autoencoder (SAE) implementation."""

    def __init__(self, input_dim, num_features):
        super().__init__(input_dim, num_features)
        self.mu = nn.Parameter(torch.zeros(input_dim))
        self.W_enc = nn.Parameter(torch.randn(num_features, input_dim) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(num_features))
        self.W_dec = nn.Parameter(torch.randn(input_dim, num_features) * 0.01)

    @torch.no_grad()
    def model_params(self):
        return {
            "input_dim": self.input_dim,
            "num_features": self.num_features,
            "mu": self.mu.detach().cpu().numpy(),
            "W_enc": self.W_enc.detach().cpu().numpy(),
            "b_enc": self.b_enc.detach().cpu().numpy(),
            "W_dec": self.W_dec.detach().cpu().numpy(),
        }

    @staticmethod
    def load_checkpoint(params: dict):
        sae = SparseLinearAE(
            input_dim=params["input_dim"], num_features=params["num_features"]
        )
        sae.mu.data.copy_(torch.from_numpy(params["mu"]))
        sae.W_enc.data.copy_(torch.from_numpy(params["W_enc"]))
        sae.b_enc.data.copy_(torch.from_numpy(params["b_enc"]))
        sae.W_dec.data.copy_(torch.from_numpy(params["W_dec"]))
        return sae

    def encode(self, x):
        return F.relu(F.linear(x - self.mu, self.W_enc, self.b_enc))

    def decode(self, z):
        return F.linear(z, self.W_dec) + self.mu

    @torch.no_grad()
    def init_from_data(self, X):
        self.mu.copy_(X.mean(dim=0))
        X_c = X - self.mu
        _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
        K = min(self.num_features, Vt.shape[0])
        self.W_dec[:, :K] = Vt[:K].T
        self.W_enc[:K] = Vt[:K]

    @torch.no_grad()
    def normalise_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)
        # Scale encoder rows to compensate
        self.W_enc.mul_(norms.T)
        self.b_enc.mul_(norms.squeeze())

    @torch.no_grad()
    def get_manifold(self, *, num_points=200, z=None):
        """Returns mu and delta for each feature, where delta is the
        scaled decoder weights."""
        mu = self.mu.detach().cpu().numpy().copy()
        dec = self.W_dec.detach().cpu().numpy().copy()

        for k in range(self.num_features):
            if z is not None:
                max_a = float(z[:, k].max())
            else:
                max_a = 1.0
            dec[:, k] *= max_a
        return mu, dec


class SparseCircularAE(AutoEncoder):
    """Sparse Circular Autoencoder (SCAE)."""

    def __init__(self, input_dim, num_features):
        super().__init__(input_dim, num_features)
        self.mu = nn.Parameter(torch.zeros(input_dim))
        # self.W_enc = nn.Parameter(
        #     torch.randn(num_features, input_dim, 2) * 0.01
        # )
        self.b_enc = nn.Parameter(torch.zeros(num_features))
        self.W_dec = nn.Parameter(
            torch.randn(input_dim, num_features, 2) * 0.01
        )

    @torch.no_grad()
    def model_params(self):
        return {
            "input_dim": self.input_dim,
            "num_features": self.num_features,
            "mu": self.mu.detach().cpu().numpy(),
            # "W_enc": self.W_enc.detach().cpu().numpy(),
            "b_enc": self.b_enc.detach().cpu().numpy(),
            "W_dec": self.W_dec.detach().cpu().numpy(),
        }

    @property
    def W_enc(self):
        """Dynamically transpose the decoder weights to get the encoder weights."""
        # Transpose from (input_dim, num_features, 2) to (num_features, input_dim, 2)
        return self.W_dec.transpose(0, 1)

    @staticmethod
    def load_checkpoint(params: dict):
        scae = SparseCircularAE(
            input_dim=params["input_dim"], num_features=params["num_features"]
        )
        scae.mu.data.copy_(torch.from_numpy(params["mu"]))
        # scae.W_enc.data.copy_(torch.from_numpy(params["W_enc"]))
        scae.b_enc.data.copy_(torch.from_numpy(params["b_enc"]))
        scae.W_dec.data.copy_(torch.from_numpy(params["W_dec"]))
        return scae

    def encode(self, x):
        xc = x - self.mu
        p = torch.einsum("bd,kdi->bki", xc, self.W_enc)  # (B, K, 2)
        p_norm = p.norm(dim=-1)
        p_unit = p / (p_norm.unsqueeze(-1) + 1e-8)
        a = F.relu(p_norm - self.b_enc)
        return a, p_unit

    def decode(self, z):
        a, p_unit = z
        d = torch.einsum("dki,bki->bkd", self.W_dec, p_unit)  # (B, K, d)
        x_hat = self.mu + (a.unsqueeze(-1) * d).sum(dim=1)  # (B, d)
        return x_hat

    def forward(self, x):
        a, p_unit = self.encode(x)
        x_hat = self.decode((a, p_unit))
        with torch.no_grad():
            angles = torch.atan2(p_unit[:, :, 1], p_unit[:, :, 0])  # (B, K)
        active = (a > 0).float()
        info = {
            "angles": angles,
            "p_unit": p_unit,
            "sparsity": active.mean().item(),
            "L0": active.sum(dim=1).mean().item(),
            "feature_activity": active.mean(dim=0).cpu().numpy(),
        }
        return x_hat, a, info

    @torch.no_grad()
    def init_from_data(self, X):
        self.mu.copy_(X.mean(dim=0))
        X_c = X - self.mu
        _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
        for k in range(self.num_features):
            i1, i2 = 2 * k, 2 * k + 1
            if i2 < Vt.shape[0]:
                self.W_enc.data[k, :, 0] = Vt[i1]
                self.W_enc.data[k, :, 1] = Vt[i2]
                self.W_dec.data[:, k, 0] = Vt[i1]
                self.W_dec.data[:, k, 1] = Vt[i2]
            # else: keep random init

    @torch.no_grad()
    def normalise_decoder(self):
        # W_dec shape: (input_dim, num_features, 2)
        # Calculate the norm over the ambient dimension (dim=0) and circular dimension (dim=2)
        norms = self.W_dec.norm(dim=(0, 2), keepdim=True).clamp(min=1e-8)

        # Normalize decoder
        self.W_dec.div_(norms)

        # W_enc shape: (num_features, input_dim, 2)
        # Transpose norms from (1, K, 1) to (K, 1, 1) to broadcast correctly
        self.W_enc.mul_(norms.transpose(0, 1))

        # b_enc shape: (num_features,)
        self.b_enc.mul_(norms.squeeze())

    @torch.no_grad()
    def get_manifold(self, *, num_points=200, z=None):
        curves = []
        for k in range(self.num_features):
            if z is not None:
                z_k = z[:, k]
                active = z_k[z_k > 0]
                if active.size > 0:
                    scale = active.mean().item()
                else:
                    scale = 1.0
            else:
                scale = 1.0

            theta = np.linspace(0, 2 * np.pi, num_points)
            phi = np.stack([np.cos(theta), np.sin(theta)], axis=1)
            W_k = self.W_dec[:, k].detach().cpu().numpy()
            loop = self.mu.detach().cpu().numpy() + scale * (phi @ W_k.T)
            curves.append(loop)
        return np.stack(curves, axis=0)
