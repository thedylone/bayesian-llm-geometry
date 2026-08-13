"""
sae_base.py — Sparse Autoencoder implementation
"""

# pylint: disable=invalid-name

from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TrainingConfig:
    """Autoencoder training hyperparameters"""

    num_epochs: int = 2000
    batch_size: int = 64
    lr: float = 1e-3
    l1_coeff: float = 0.01
    normalise_every: int = 100
    device: torch.device = None


class AutoEncoder(nn.Module, ABC):
    """Base class for autoencoders.

    Args:
        input_dim: Ambient dimension of inputs
        num_features: Number of features (latent dimension)
    """

    def __init__(self, input_dim: int, num_features: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features

    @abstractmethod
    def model_params(self) -> dict:
        """Return model parameters.

        Returns:
            Dictionary of model parameters.
        """

    @staticmethod
    @abstractmethod
    def load_checkpoint(params: dict) -> "AutoEncoder":
        """Create a model instance from saved parameters.

        Args:
            params: Dictionary of model parameters.
        Returns:
            AutoEncoder: A new instance of the AutoEncoder class.
        """

    def forward(self, x: torch.Tensor):
        """Encode and decode input data.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            (x_recon, z, info): Reconstructed tensor, latent features,
            and info dict (sparsity, L0, feature_activity)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        active = (z > 0).float()
        info = {
            "sparsity": active.mean().item(),
            "L0": active.sum(dim=1).mean().item(),
            "feature_activity": active.mean(dim=0).cpu().numpy(),
        }
        return x_recon, z, info

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input data to latent features.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            z: Latent features of shape (batch_size, num_features)
        """

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent features to reconstruct input data.

        Args:
            z: Latent features of shape (batch_size, num_features)

        Returns:
            x_recon: Reconstructed tensor of shape (batch_size, input_dim)
        """

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        l1_coeff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SAE loss: MSE reconstruction + L1 sparsity penalty.

        Args:
            x: Original input tensor, shape (batch_size, input_dim)
            x_hat: Reconstructed tensor, shape (batch_size, input_dim)
            z: Latent activations, shape (batch_size, num_features)
            l1_coeff: Coefficient for L1 sparsity penalty

        Returns:
            loss: Total loss (MSE + L1)
            mse: Mean squared error component
            l1: L1 sparsity penalty component
        """
        mse = F.mse_loss(x_hat, x, reduction="mean")
        # l1 = torch.mean(torch.abs(z))
        l1 = z.abs().sum(dim=1).mean()  # L1 penalty on activations
        loss = mse + l1_coeff * l1
        return loss, mse, l1

    @abstractmethod
    def init_from_data(self, X: torch.Tensor) -> None:
        """Initialise model parameters from data using PCA.

        Args:
            X: Data tensor of shape (N, d)
        """

    @abstractmethod
    def normalise_decoder(self) -> None:
        """Normalise decoder weights to unit norm."""

    def train_model(
        self,
        X_train: torch.Tensor,
        config: TrainingConfig = None,
        *,
        pbar_pos=0,
        pbar_leave=True,
    ) -> dict[str, list]:
        """Train on data.

        Args:
            self:            AutoEncoder class
            X_train:         Training data as numpy array, shape (N, d).
            num_epochs:      Number of training epochs.
            batch_size:      Mini-batch size.
            lr:              Learning rate.
            l1_coeff:        L1 sparsity penalty coefficient.
            normalise_every: How often to normalise decoder/subspaces.
                    Set <= 0 or None to disable normalisation.
            device:          torch.device or string (e.g., 'cuda', 'cpu').
                            If None, auto-selects GPU if available.

        Returns:
            Dict with training history: losses, mse, l1, sparsity.
        """
        if config is None:
            config = TrainingConfig()
        config.device = self._get_device(config.device)
        self.to(config.device)
        X_t = torch.tensor(X_train, dtype=torch.float32).to(config.device)
        self.init_from_data(X_t)

        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimiser = torch.optim.Adam(self.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=config.num_epochs, eta_min=config.lr * 0.01
        )

        return self._train(
            loader,
            optimiser,
            scheduler,
            config,
            pbar_pos=pbar_pos,
            pbar_leave=pbar_leave,
        )

    def _train(
        self,
        loader,
        optimiser,
        scheduler,
        config,
        *,
        pbar_pos=0,
        pbar_leave=True,
    ) -> dict[str, list]:
        """Default training loop."""
        history = {"loss": [], "mse": [], "l1": [], "sparsity": []}
        pbar = tqdm(
            range(config.num_epochs),
            desc="Training",
            unit="epoch",
            position=pbar_pos,
            leave=pbar_leave,
        )

        for epoch in pbar:
            epoch_loss, epoch_mse, epoch_l1, epoch_sp = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(config.device)
                x_hat, a, info = self(batch)
                loss, mse, l1 = self.loss(batch, x_hat, a, config.l1_coeff)

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimiser.step()

                epoch_loss += loss.item()
                epoch_mse += mse.item()
                epoch_l1 += l1.item()
                epoch_sp += info["sparsity"]
                n_batches += 1

            scheduler.step()

            if (
                config.normalise_every
                and (epoch + 1) % config.normalise_every == 0
            ):
                self.normalise_decoder()

            self._update_history(
                history, epoch_loss, epoch_mse, epoch_l1, epoch_sp, n_batches
            )
            pbar.set_postfix(
                {
                    "loss": f"{history['loss'][-1]:.5f}",
                    "mse": f"{history['mse'][-1]:.5f}",
                    "l1": f"{history['l1'][-1]:.5f}",
                    "sparsity": f"{history['sparsity'][-1]:.2%}",
                }
            )

        return history

    def _update_history(
        self, history, epoch_loss, epoch_mse, epoch_l1, epoch_sp, n_batches
    ) -> None:
        """Update training history."""
        history["loss"].append(epoch_loss / n_batches)
        history["mse"].append(epoch_mse / n_batches)
        history["l1"].append(epoch_l1 / n_batches)
        history["sparsity"].append(epoch_sp / n_batches)

    def _get_device(self, device) -> torch.device:
        """Get the appropriate device."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def evaluate(self, X_test: torch.Tensor, device=None):
        """Evaluate model on test data.

        Args:
            X_test: Test data tensor of shape (N, d).
            device: torch.device or string (e.g., 'cuda', 'cpu').
                    If None, auto-selects GPU if available.

        Returns:
            Dict with evaluation metrics:
        """
        device = self._get_device(device)

        self.to(device)
        X_t = X_test.to(device)

        with torch.no_grad():
            x_hat, z, info = self.forward(X_t)
            loss, mse, l1 = self.loss(X_t, x_hat, z, l1_coeff=0.01)

        return {
            "z": z.cpu().numpy(),
            "loss": loss.item(),
            "mse": mse.item(),
            "l1": l1.item(),
            "info": info,
        }

    def evaluate_np(self, X_test: np.ndarray, device=None):
        """Evaluate model on test data (numpy array).

        Args:
            X_test: Test data numpy array of shape (N, d).
            device: torch.device or string (e.g., 'cuda', 'cpu').
                    If None, auto-selects GPU if available.

        Returns:
            Dict with evaluation metrics.
        """
        X_t = torch.tensor(X_test, dtype=torch.float32)
        return self.evaluate(X_t, device=device)

    @abstractmethod
    def get_manifold(self, *, num_points=200, z=None):
        """Get points along the fitted manifold per feature

        Args:
            num_points: Discretisation of manifold curve per feature
            z: Latent activations to use for manifold computation.

        Returns:
            Points (num_features, num_points, input_dim)
        """
