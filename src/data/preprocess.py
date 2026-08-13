"""
process.py — Process LLM activations for analysis.
"""

# pylint: disable=invalid-name

import numpy as np


def subtract_template_mean(
    X: np.ndarray, template_ids: np.ndarray
) -> np.ndarray:
    """Remove per-template common-mode variance from activations.

    For each template t, compute the mean activation of the rows with
    that template_id (averaging over all labels) and subtract it from
    each row. The global mean is then re-added so the data stays
    centred at its original location.

    This is the standard fix when template variance dominates the top
    PCs of the raw activations — template common-mode appears in the
    grand mean within each template group and gets removed, while the
    label-distinguishing signal (which varies across labels for a
    fixed template) is preserved.

    Args:
        X:            (N, d) raw activations.
        template_ids: (N,) per-row template index.

    Returns:
        (N, d) cleaned activations, same dtype as X.
    """
    X_clean = X.copy()
    for t in np.unique(template_ids):
        mask = template_ids == t
        X_clean[mask] -= X[mask].mean(axis=0, keepdims=True)
    X_clean += X.mean(axis=0, keepdims=True)
    return X_clean


def pca_reduce_activations(
    X: np.ndarray,
    n_components: int = 20,
) -> np.ndarray:
    """Mean-centre and PCA-reduce LLM activations.

    Args:
        X:            (N, d) activation matrix (d = d_model).
        n_components: Number of PCs to retain (default 20, matching the
                      m=20 synthetic validation regime).

    Returns:
        (N, n_components) float32 array.
    """
    mu = X.mean(axis=0)
    X_c = X - mu

    # Thin SVD: X_c = U diag(S) Vt, with S sorted descending.
    # Scores (projections onto top-k PCs) are X_c @ Vt[:k].T = U[:, :k] * S[:k]
    U, S, _ = np.linalg.svd(X_c, full_matrices=False)

    k = n_components
    scores = U[:, :k] * S[:k]  # (N, k), variance per component = S_i^2 / N

    var_explained = (S[:k] ** 2).sum() / (S**2).sum()
    print(f"  PCA ({k} components): {var_explained:.1%} variance retained")

    return scores.astype(np.float64)


def preprocess_activations(
    X: np.ndarray,
    template_ids: np.ndarray,
    pca_components: int = 20,
) -> np.ndarray:
    """Preprocess LLM activations for analysis.
    Performs template-mean subtraction and PCA reduction.

    Args:
        X:              (N, d) activation matrix (d = d_model).
        template_ids:   (N,) per-row template index.
        pca_components: Number of PCA components to retain (default 20).

    Returns:
        (N, pca_components) preprocessed activations.
    """
    if template_ids is not None:
        X = subtract_template_mean(X, template_ids)
    return pca_reduce_activations(X, n_components=pca_components)
