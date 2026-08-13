"""
models.py — Bayesian Model Comparison implementations

EM inference for both generative models, plus Laplace-approximated
marginal likelihood for Bayes factor computation.

Models:
  ML (Linear):   z ~ N(0,1),  x|z ~ N(mu + w*z, sigma^2 I)
  MC (Circular): z ~ Uniform(0,2pi),
                 x|z ~ N(mu + W @ [cos(z), sin(z)], sigma^2 I)
"""

# pylint: disable=invalid-name
# Mathematical convention: uppercase for matrices (X, W, N), lowercase for
# vectors (w, z, mu). Disabling invalid-name globally for readability.

import numpy as np
from src.bmc.base import Model


class LinearModel(Model):
    """Linear generative model:
    - z ~ N(0,1)
    - x|z ~ N(mu + w*z, sigma^2 I)

    Parameters:
        X (np.ndarray): Activations, shape (N, m)
    """

    def n_params(self, m=None):
        """2m + 1: mu (m) + w (m) + sigma2 (1)."""
        if m is None:
            raise ValueError("m must be specified")
        return 2 * m + 1

    def log_likelihood(self, X, params):
        """Log-likelihood of the linear model.

        log p(X|θ,M) = sum_n log N(x_n | mu, sigma^2 I + w w^T)

        Uses the matrix determinant lemma and Woodbury identity to avoid
        forming the full m x m covariance, reducing cost from O(Nm² + m³)
        to O(Nm).

        Determinant:  |σ²I + ww^T| = σ^{2m} (1 + w^Tw / σ²)
        Inverse:      (σ²I + ww^T)^{-1} = (1/σ²)(I - ww^T / (σ² + w^Tw))
        """
        mu, w, sigma2 = params["mu"], params["w"], params["sigma2"]
        N, m = X.shape
        X_centered = X - mu

        wtw = np.dot(w, w)  # scalar
        log_det = m * np.log(sigma2) + np.log1p(wtw / sigma2)

        # Quadratic forms via Woodbury:
        # x^T Σ^{-1} x = (1/σ²)(||x||² - (w^T x)² / (σ² + w^T w))
        proj = X_centered @ w  # (N,)
        quad = (
            np.sum(X_centered**2) - np.sum(proj**2) / (sigma2 + wtw)
        ) / sigma2

        ll = -0.5 * (N * (log_det + m * np.log(2 * np.pi)) + quad)
        return ll

    def closed_form_mle(self, X):
        """Closed-form MLE parameters for linear model."""
        N, _ = X.shape
        mu = np.mean(X, axis=0)
        X_centered = X - mu
        cov = X_centered.T @ X_centered / N
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        w = eigvecs[:, idx[0]] * np.sqrt(
            np.max(eigvals[idx[0]] - np.mean(eigvals[idx[1:]]), 0)
        )
        sigma2 = np.mean(eigvals[idx[1:]])
        return {"mu": mu, "w": w, "sigma2": sigma2}

    def log_evidence_laplace(self, X, params):
        """Compute Laplace approximation to log-evidence.

        log p(X|M) ≈ log p(X|θ̂,M) + (k/2) log(2π) - 0.5 log det(H)
        where H is the Hessian of the negative log-likelihood at θ̂.
        """
        mu, w, sigma2 = params["mu"], params["w"], params["sigma2"]
        log_sigma2 = np.log(sigma2)
        ll = self.log_likelihood(X, params)
        lp = self.log_prior(mu, w, log_sigma2)

        theta = np.concatenate([mu, w, [0.5 * log_sigma2]])

        def neg_log_post(theta):
            mu_ = theta[: len(mu)]
            w_ = theta[len(mu) : len(mu) + len(w)]  # noqa: E203
            log_sigma_ = theta[-1]
            sigma2_ = np.exp(2 * log_sigma_)
            ll_ = self.log_likelihood(
                X, {"mu": mu_, "w": w_, "sigma2": sigma2_}
            )
            lp_ = self.log_prior(mu_, w_, 2 * log_sigma_)
            return -ll_ - lp_

        H = self._diag_hessian(neg_log_post, theta)
        k = len(theta)
        log_det_H = np.sum(np.log(H))

        return ll + lp + 0.5 * k * np.log(2 * np.pi) - 0.5 * log_det_H

    def _em(self, X, max_iter=1000, tol=1e-8, verbose=False):
        """Fit the linear model via EM.

        Returns:
            dict: MLE parameters and log-likelihood history.
        """
        # Initialise parameters
        mu: np.ndarray = np.mean(X, axis=0)
        X_centered = X - mu
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        w: np.ndarray = Vt[0] * np.std(X_centered @ Vt[0])
        sigma2: float = np.mean(np.var(X_centered, axis=0))

        log_mls = []
        for it in range(max_iter):
            # E-step: compute posterior over z
            denom = np.dot(w, w) + sigma2
            sigma_z2 = sigma2 / denom  # Var[z|x]
            z_hat = X_centered @ w / denom  # E[z|x]

            # M-step: update parameters sequentially
            mu = np.mean(X - z_hat[:, None] * w[None, :], axis=0)
            X_centered = X - mu
            sum_expected_z2 = np.sum(z_hat**2 + sigma_z2)
            w = (X_centered.T @ z_hat) / sum_expected_z2
            sigma2 = (
                np.sum(X_centered**2)
                - 2 * np.dot(z_hat, X_centered @ w)
                + sum_expected_z2 * np.dot(w, w)
            ) / (X.size)
            sigma2 = max(sigma2, 1e-10)

            ll = self.log_likelihood(X, {"mu": mu, "w": w, "sigma2": sigma2})
            log_mls.append(ll)

            if verbose and (it + 1) % 10 == 0:
                print(
                    f"  EM iter {it + 1:4d}: "
                    f"||μ||={np.linalg.norm(mu):.4f}, "
                    f"||w||={np.linalg.norm(w):.4f}, "
                    f"σ^2={sigma2:.4f}, "
                    f"log p(X|θ)={ll:.2f}"
                )
            if it > 0 and abs(log_mls[-1] - log_mls[-2]) < tol:
                if verbose:
                    print(f" Converged: iter {it + 1}, log p(X|θ)={ll:.2f}")
                break
        return {"mu": mu, "w": w, "sigma2": sigma2, "log_mls": log_mls}

    def manifold_samples(self, params, n_pts=100):
        """Generate samples from the linear manifold given parameters."""
        t = np.linspace(-3, 3, n_pts)
        return params["mu"] + t[:, None] * params["w"]


class CircularModel(Model):
    """Circular generative model:
    - z ~ Uniform(0,2pi)
    - x|z ~ N(mu + W @ [cos(z), sin(z)], sigma^2 I)

    Parameters:
        X (np.ndarray): Activations, shape (N, m)
        K (int): Number of grid points for numerical integration over z
    """

    def n_params(self, m=None):
        """3m + 1: mu (m) + W (2m) + sigma2 (1)."""
        if m is None:
            raise ValueError("m must be specified")
        return 3 * m + 1

    def __init__(self, K=200):
        super().__init__()
        self.K = K
        """number of grid points for numerical integration over z"""
        # Precompute grid
        self.grid = np.linspace(0, 2 * np.pi, K, endpoint=False)
        self.dz = self.grid[1] - self.grid[0]
        self.phi_grid = np.stack(
            [np.cos(self.grid), np.sin(self.grid)], axis=1
        )  # (K, 2)

    def log_likelihood(self, X, params):
        """Log-likelihood of the circular model.

        log p(X|θ,M) = sum_n log[1/K*sum_k N(x_n; mu + W*phi(z^k), sigma^2 I)]
        where K uniform grid points numerically integrate over z ~ U(0,2pi).
        """
        mu, W, sigma2 = params["mu"], params["W"], params["sigma2"]
        _, m = X.shape
        X_centered = X - mu
        # residuals[n, k] = x_n - mu - W @ phi(z^k), shape (N, K, m)
        residuals = X_centered[:, None, :] - (self.phi_grid @ W.T)[None, :, :]
        log_N_nk = -0.5 * (
            m * np.log(2 * np.pi * sigma2)
            + np.sum(residuals**2, axis=2) / sigma2
        )  # (N, K)
        # log [1/K * sum_k exp(log_N_nk)] = logsumexp - log(K)
        log_p_n = (
            np.log(
                np.sum(
                    np.exp(log_N_nk - log_N_nk.max(axis=1, keepdims=True)),
                    axis=1,
                )
            )
            + log_N_nk.max(axis=1)
            - np.log(self.K)
        )
        return np.sum(log_p_n)

    def log_evidence_laplace(self, X, params):
        """Compute Laplace approximation to log-evidence.

        log p(X|M) ≈ log p(X|θ̂,M) + log p(θ̂|M)\
                    + (k/2) log(2π) - (1/2) log det(H)
        where H is the Hessian of the negative log-posterior at θ̂.
        θ = (mu, W.flatten(), log_sigma), k = m + 2m + 1 = 3m + 1
        """
        mu, W, sigma2 = params["mu"], params["W"], params["sigma2"]
        log_sigma2 = np.log(sigma2)
        ll = self.log_likelihood(X, params)
        lp = self.log_prior(mu, W, log_sigma2)

        theta = np.concatenate([mu, W.flatten(), [0.5 * log_sigma2]])

        def neg_log_post(theta):
            m = len(mu)
            mu_ = theta[:m]
            W_ = theta[m : m + 2 * m].reshape(m, 2)  # noqa: E203
            log_sigma_ = theta[-1]
            sigma2_ = np.exp(2 * log_sigma_)
            ll_ = self.log_likelihood(
                X, {"mu": mu_, "W": W_, "sigma2": sigma2_}
            )
            lp_ = self.log_prior(mu_, W_, 2 * log_sigma_)
            return -ll_ - lp_

        H = self._diag_hessian(neg_log_post, theta)
        k = len(theta)
        log_det_H = np.sum(np.log(H))

        return ll + lp + 0.5 * k * np.log(2 * np.pi) - 0.5 * log_det_H

    def _e_step(self, X_centered, W, sigma2):
        """E-step: compute grid posteriors and sufficient statistics.

        Returns:
            E_phi:     E[phi(z_n)],          shape (N, 2)
            E_phiphi:  E[phi(z_n)phi(z_n)^T], shape (N, 2, 2)
        """
        # log p(z^k | x_n) ∝ -1/(2σ²) ||x_n - μ - W φ(z^k)||²
        residuals = (
            X_centered[:, None, :] - (self.phi_grid @ W.T)[None, :, :]
        )  # (N, K, m)
        log_p_nk = -0.5 / sigma2 * np.sum(residuals**2, axis=2)  # (N, K)
        # Normalise with log-sum-exp for numerical stability
        log_p_nk -= np.log(
            np.sum(
                np.exp(log_p_nk - log_p_nk.max(axis=1, keepdims=True)),
                axis=1,
                keepdims=True,
            )
        ) + log_p_nk.max(axis=1, keepdims=True)
        p_nk = np.exp(log_p_nk)  # (N, K)
        # Sufficient statistics
        E_phi = p_nk @ self.phi_grid  # (N, 2)
        E_phiphi = np.einsum(
            "nk,ki,kj->nij", p_nk, self.phi_grid, self.phi_grid
        )  # (N, 2, 2)
        return E_phi, E_phiphi

    def _em(self, X, max_iter=1000, tol=1e-8, verbose=False):
        """Fit the circular model via EM.

        Returns:
            dict: MLE parameters and log-likelihood history.
        """
        N, m = X.shape

        # Initialise parameters
        mu = np.mean(X, axis=0)
        X_centered = X - mu
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        W = Vt[:2].T * np.std(
            X_centered @ Vt[:2].T, axis=0, keepdims=True
        )  # (m, 2)
        sigma2 = np.mean(np.var(X_centered, axis=0))

        log_mls = []
        for it in range(max_iter):
            # E-step: compute sufficient statistics
            E_phi, E_phiphi = self._e_step(X_centered, W, sigma2)
            # (N,2), (N,2,2)

            # M-step: sequential updates
            mu = np.mean(X - E_phi @ W.T, axis=0)
            X_centered = X - mu
            sum_E_phiphi = np.sum(E_phiphi, axis=0)  # (2, 2)
            W = (X_centered.T @ E_phi) @ np.linalg.inv(sum_E_phiphi)  # (m, 2)

            # σ² = (1/Nm) Σ_n E[||x_n - μ - Wφ(z_n)||²]
            # expanded: ||x_n-μ||² - 2(x_n-μ)^T W E[φ] + tr(W^T W E[φφ^T])
            WtW = W.T @ W  # (2, 2)
            expected_quad = (
                np.sum(X_centered**2)
                - 2 * np.sum(X_centered * (E_phi @ W.T))
                + np.sum(WtW * sum_E_phiphi)
            )
            sigma2 = max(expected_quad / (N * m), 1e-10)

            ll = self.log_likelihood(X, {"mu": mu, "W": W, "sigma2": sigma2})
            log_mls.append(ll)

            if verbose and (it + 1) % 10 == 0:
                print(
                    f"  EM iter {it + 1:4d}: "
                    f"||μ||={np.linalg.norm(mu):.4f}, "
                    f"||W||={np.linalg.norm(W):.4f}, "
                    f"σ^2={sigma2:.4f}, "
                    f"log p(X|θ)={ll:.2f}"
                )
            if it > 0 and abs(log_mls[-1] - log_mls[-2]) < tol:
                if verbose:
                    print(f"  Converged: iter {it + 1}, log p(X|θ)={ll:.2f}")
                break

        return {"mu": mu, "W": W, "sigma2": sigma2, "log_mls": log_mls}

    def manifold_samples(self, params, n_pts=100):
        theta = np.linspace(0, 2 * np.pi, n_pts)
        phi = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        return params["mu"] + phi @ params["W"].T
