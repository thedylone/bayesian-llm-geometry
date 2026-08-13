"""
synthetic.py — Synthetic data generators shared across plotting scripts.
"""

# pylint: disable=invalid-name
import os
import numpy as np


def generate_for_seed(seed, n_samples, dim, sigma2, out_dir):
    """Generate synthetic data for a specific seed."""
    suffix = f"d{dim}_s{seed}.npz"
    # Generate linear synthetic data and save
    linear_file_path = f"{out_dir}/linear_data_{suffix}"
    if not os.path.exists(linear_file_path):
        linear_data, linear_truth = generate_linear(
            n_samples=n_samples,
            dim=dim,
            seed=seed,
            sigma2=sigma2,
        )
        np.savez(
            linear_file_path,
            activations=linear_data,
            mu=linear_truth["mu"],
            w=linear_truth["w"],
            sigma2=linear_truth["sigma2"],
            z=linear_truth["z"],
        )
    # Generate circular synthetic data and save
    circular_file_path = f"{out_dir}/circular_data_{suffix}"
    if not os.path.exists(circular_file_path):
        circular_data, circular_truth = generate_circular(
            n_samples=n_samples,
            dim=dim,
            seed=seed,
            sigma2=sigma2,
        )
        np.savez(
            circular_file_path,
            activations=circular_data,
            mu=circular_truth["mu"],
            w=circular_truth["w"],
            sigma2=circular_truth["sigma2"],
            z=circular_truth["z"],
        )


def generate_linear(n_samples=200, dim=20, seed=42, sigma2=0.05):
    """Generate data from the linear generative model."""
    rng = np.random.default_rng(seed)
    mu_true = rng.standard_normal(dim) * 0.5
    w_true = rng.standard_normal(dim)
    w_true = w_true / np.linalg.norm(w_true) * 2.0

    z = rng.standard_normal(n_samples)
    noise = rng.normal(0, np.sqrt(sigma2), (n_samples, dim))
    X = mu_true + z[:, None] * w_true + noise

    truth = {"mu": mu_true, "w": w_true, "sigma2": sigma2, "z": z}
    return X, truth


def generate_circular(n_samples=200, dim=20, seed=42, sigma2=0.05):
    """Generate data from the circular generative model."""
    rng = np.random.default_rng(seed)
    mu_true = rng.standard_normal(dim) * 0.5
    W_true = rng.standard_normal((dim, 2))
    U, _, Vt = np.linalg.svd(W_true, full_matrices=False)
    W_true = U @ np.diag([2.0, 2.0]) @ Vt

    z = rng.uniform(0, 2 * np.pi, n_samples)
    phi = np.stack([np.cos(z), np.sin(z)], axis=1)
    noise = rng.normal(0, np.sqrt(sigma2), (n_samples, dim))
    X = mu_true + phi @ W_true.T + noise

    truth = {"mu": mu_true, "w": W_true, "sigma2": sigma2, "z": z}
    return X, truth


def generate_helical(
    n_samples=200,
    dim=20,
    seed=42,
    helix_spacing=1.0,
    total_angle=4 * np.pi,
    sigma2=0.05,
):
    """
    Generate data from a helical generative model.

    The latent variable z is drawn uniformly over [0, total_angle].
    Each sample is mapped to a point on a helix in a 3D latent space:

        phi(z) = [cos(z), sin(z), z * helix_spacing / (2 * pi)]

    and then projected into the ambient dim-space via an orthogonal
    projection matrix W (dim x 3), following the same construction
    as generate_circular.

    Parameters
    ----------
    n_samples     : int   - number of data points
    dim           : int   - ambient dimension (must be >= 3)
    seed          : int   - RNG seed
    helix_spacing : float - axial rise per full revolution (2π radians).
                            Larger values → more stretched-out helix.
    total_angle   : float - total angular sweep in radians.
                            E.g. 2π = one loop, 4π = two loops.
    """
    assert dim >= 3, "Helical model requires dim >= 3"

    rng = np.random.default_rng(seed)
    mu_true = rng.standard_normal(dim) * 0.5

    # Orthonormal projection matrix: (dim x 3), scales set to 2.0
    W_raw = rng.standard_normal((dim, 3))
    U, _, Vt = np.linalg.svd(W_raw, full_matrices=False)
    W_true = U @ np.diag([2.0, 2.0, 2.0]) @ Vt  # (dim x 3)

    # Latent variable: angle along helix
    z = rng.uniform(0, total_angle, n_samples)

    # Helix embedding: axial component normalised so
    # one full turn = helix_spacing
    axial = z * helix_spacing / (2 * np.pi)
    phi = np.stack([np.cos(z), np.sin(z), axial], axis=1)  # (n_samples x 3)

    noise = rng.normal(0, np.sqrt(sigma2), (n_samples, dim))
    X = mu_true + phi @ W_true.T + noise  # (n_samples x dim)

    truth = {
        "mu": mu_true,
        "w": W_true,
        "sigma2": sigma2,
        "z": z,
        "helix_spacing": helix_spacing,
        "total_angle": total_angle,
    }
    return X, truth
