"""Helpers for plotting BMC manifolds."""

import numpy as np
import matplotlib.pyplot as plt

from src.vis.plot_helpers import (
    C_LINEAR,
    C_CIRCULAR,
    plot_activations,
    label_colours,
    annotate_text,
)


def linear_manifold_curve(params, n_pts=200):
    """Compute points along the fitted linear manifold.

    Args:
        params: Fitted parameter dict with keys 'mu' and 'w'.
        n_pts: Number of points to sample along the line.

    Returns:
        Array of shape (n_pts, d).
    """
    t = np.linspace(-3, 3, n_pts)
    return params["mu"] + t[:, None] * params["w"]


def circular_manifold_curve(params, n_pts=300):
    """Compute points along the fitted circular manifold.

    Args:
        params: Fitted parameter dict with keys 'mu' and 'W'
                (where 'W' is the d×2 matrix W for the circular model).
        n_pts: Number of points to sample around the circle.

    Returns:
        Array of shape (n_pts, d).
    """
    theta = np.linspace(0, 2 * np.pi, n_pts)
    phi = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return params["mu"] + phi @ params["W"].T


def add_manifolds_to_ax(ax, linear_params, circular_params):
    """Add fitted manifolds to an axis.

    Args:
        ax: Matplotlib axis.
        linear_params: Fitted parameters for the linear model.
        circular_params: Fitted parameters for the circular model.
    """
    lin_manifold = linear_manifold_curve(linear_params)
    circ_manifold = circular_manifold_curve(circular_params)
    ndim = 3 if ax.name == "3d" else 2

    for manifold, color, label in [
        (lin_manifold, C_LINEAR, r"$M_L$ (linear) fit"),
        (circ_manifold, C_CIRCULAR, r"$M_C$ (circular) fit"),
    ]:
        ax.plot(*manifold[:, :ndim].T, color=color, lw=5, label=label)


def annotate_bf(ax, log_bf):
    """Add a Bayes factor annotation to the middle of an axis.

    Args:
        ax: Matplotlib axis.
        log_bf: Log Bayes factor value.
    """
    verdict = "$M_L$" if log_bf > 0 else "$M_C$"
    text = rf"$\log \mathrm{{BF}}$ = {log_bf:+.1f} ({verdict})"
    annotate_text(ax, text)


def plot_bmc_scatter(
    activations,
    bmc_params,
    *,
    labels=None,
    ax=None,
    short_labels=None,
):
    """Plot BMC results.

    Args:
        ax: Matplotlib axis (3D or 2D, default 3D).
        activations: (N, d) array of LLM activations.
        labels: Array of labels for the activations.
        bmc_params: Dictionary containing BMC parameters.

    Returns:
        ax: Matplotlib axis.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    if labels is not None:
        label_cols = label_colours(labels)
        colours = np.array([label_cols[label] for label in labels])
    else:
        label_cols = None
        colours = None
    # Scatter activations
    plot_activations(
        activations,
        ax=ax,
        labels=labels,
        colours=colours,
        label_cols=label_cols,
        short_labels=short_labels,
    )
    # Add fitted manifolds
    add_manifolds_to_ax(ax, bmc_params["linear"], bmc_params["circular"])
    annotate_bf(ax, bmc_params["log_bf"])

    return ax


def plot_bmc_training(linear_log_mls, circular_log_mls, *, ax=None):
    """Plot the training log marginal likelihoods for BMC.

    Args:
        linear_log_mls: Array of log marginal likelihoods for linear model.
        circular_log_mls: Array of log marginal likelihoods for circular model.
        ax: Optional Matplotlib axis to plot on.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(
        linear_log_mls,
        label="Linear model",
        color=C_LINEAR,
        lw=3,
    )
    ax.plot(
        circular_log_mls,
        label="Circular model",
        color=C_CIRCULAR,
        lw=3,
    )
    ax.set_xlabel("EM iteration")
    ax.set_ylabel(r"$\log\, p(\mathbf{X} \mid \hat{\theta})$")
    ax.legend()
    return ax
