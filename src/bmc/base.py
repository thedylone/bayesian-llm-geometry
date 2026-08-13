"""
base.py — Base class for generative models

EM inference for both generative models, plus Laplace-approximated
marginal likelihood for Bayes factor computation.
"""

# pylint: disable=invalid-name
# Mathematical convention: uppercase for matrices (X, W, N), lowercase for
# vectors (w, z, mu). Disabling invalid-name globally for readability.


import numpy as np


class Model:
    """Base class for generative models. Subclasses must implement:
    - log_likelihood(params): log p(X|θ,M)
    - _em(): EM algorithm to find MLE parameters
    - log_evidence_laplace(params): Laplace approximation to log p(X|M)
    """

    def __init__(self):
        self.params_mle = None
        """MLE parameters dict, set after fit()"""
        self.log_evidence = None
        """log-evidence, set after fit()"""

    def __repr__(self):
        return self.__class__.__name__

    def n_params(self, m=None):
        """Number of free parameters in the model."""
        raise NotImplementedError

    @staticmethod
    def log_prior(mu, w, log_sigma2):
        """Log-prior over parameters.
        Weakly informative Gaussian priors on mu, w, log(sigma^2)."""
        return -0.5 / 100 * (np.sum(mu**2) + np.sum(w**2) + log_sigma2**2)

    def log_likelihood(self, X, params):
        """Log-likelihood of the model given parameters."""
        raise NotImplementedError

    def log_evidence_laplace(self, X, params):
        """Compute Laplace approximation to log-evidence."""
        raise NotImplementedError

    def log_evidence_bic(self, X, params):
        """BIC approximation to log-evidence.

        log p(X|M) ≈ log p(X|θ̂,M) - (k/2) log(N)

        Simpler than Laplace: replaces the Hessian-based Occam penalty
        with a flat penalty of (k/2) log(N) per parameter. More robust
        when the posterior is far from Gaussian.
        """
        N = X.shape[0]
        k = self.n_params(m=X.shape[1])
        ll = self.log_likelihood(X, params)
        return ll - 0.5 * k * np.log(N)

    @staticmethod
    def _diag_hessian(f, theta, eps=1e-4):
        """Diagonal Hessian via central finite differences."""
        k = len(theta)
        f0 = f(theta)
        H = np.zeros(k)
        for i in range(k):
            e = np.zeros(k)
            e[i] = eps
            H[i] = (f(theta + e) - 2 * f0 + f(theta - e)) / eps**2
        return np.maximum(H, 1e-6)

    def fit(self, X: np.ndarray, max_iter=1000, tol=1e-8, verbose=False):
        """Fit the model parameters via EM and store MLE parameters.

        Args:
            X (np.ndarray): Activations, shape (N, m)
            max_iter (int): Maximum number of EM iterations.
            tol (float): Convergence tolerance for log-likelihood.
            verbose (bool): If True, print EM progress.

        Returns:
            dict: MLE parameters and log-likelihood history.
        """
        self.params_mle = self._em(
            X, max_iter=max_iter, tol=tol, verbose=verbose
        )
        self.log_evidence = self.log_evidence_laplace(X, self.params_mle)
        return self.params_mle

    def _em(self, X, max_iter=1000, tol=1e-8, verbose=False):
        """EM algorithm to find MLE parameters
        Returns:
            dict: MLE parameters and log-likelihood history.
        """
        raise NotImplementedError

    def manifold_samples(self, params, n_pts=100):
        """Generate samples from the model's manifold given parameters."""
        raise NotImplementedError
