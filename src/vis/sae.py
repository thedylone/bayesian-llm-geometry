"""SAE-specific plotting helpers."""

import numpy as np
import matplotlib.pyplot as plt
from src.sae.models import SparseLinearAE, SparseCircularAE
from src.vis.plot_helpers import (
    C_SAE,
    C_SCAE,
    plot_activations,
    label_colours,
    annotate_text,
    get_short_label,
)


def feature_colour_map(alive_idx):
    """Assign a distinct colour to each alive feature index."""
    cmap = plt.get_cmap("tab10" if len(alive_idx) <= 10 else "tab20")
    n = max(cmap.N, 1)
    return {int(k): cmap(i % n) for i, k in enumerate(alive_idx)}


def dominant_feature_colours(
    act_np, alive_idx, feat_colours, inactive_rgba=(0.72, 0.72, 0.75, 0.35)
):
    """Colour each point by its argmax-active feature."""
    dom = act_np.argmax(axis=1)
    max_a = act_np.max(axis=1)
    alive_set = set(int(k) for k in alive_idx)
    colors = []
    for n in range(act_np.shape[0]):
        if max_a[n] <= 0 or int(dom[n]) not in alive_set:
            colors.append(inactive_rgba)
        else:
            colors.append(feat_colours[int(dom[n])])
    return np.array(colors)


def plot_atom_vectors(
    sae: SparseLinearAE, alive_idx, feat_cols, z=None, ax=None
):
    """Plot the atom vectors of a fitted SAE model."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ndim = 3 if ax.name == "3d" else 2
    mu, delta = sae.get_manifold(z=z)

    # Set the formatting arguments specific to the axis dimension
    if ndim == 3:
        quiver_kwargs = {"arrow_length_ratio": 0.15}
    else:
        quiver_kwargs = {"angles": "xy", "scale_units": "xy", "scale": 1}

    for k in alive_idx:
        # Unpack both the positions (mu) and the directions (delta)
        ax.quiver(
            *mu[:ndim],
            *delta[:ndim, k],
            color=feat_cols[k],
            lw=3,
            **quiver_kwargs,
        )

    return ax


def plot_atom_loops(
    scae: SparseCircularAE, alive_idx, feat_cols, z=None, ax=None
):
    """Plot the atom loops of a fitted SCAE model in 2D or 3D."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ndim = 3 if ax.name == "3d" else 2
    loops = scae.get_manifold(z=z)

    for k in alive_idx:
        ax.plot(
            *loops[k][:, :ndim].T,
            color=feat_cols[k],
            lw=3,
        )

    return ax


def annotate_eval(ax, evaluation):
    """Add evaluation metrics to the middle of an axis.

    Args:
        ax: Matplotlib axis.
        evaluation: Dict with evaluation metrics.
    """
    text = f"MSE: {evaluation['mse']:.2f}, L1: {evaluation['l1']:.2f}"
    annotate_text(ax, text)


def plot_fitted_sae(
    activations,
    labels,
    model,
    evaluation,
    *,
    alive_threshold=0.1,
    ax=None,
    short_labels=None,
):
    """Plot data and fitted atoms for an SAE-like model."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    # Get alive features
    feat_activity = evaluation["info"]["feature_activity"]
    alive_idx = np.where(feat_activity > alive_threshold)[0]
    feat_cols = feature_colour_map(alive_idx)
    z = evaluation.get("z")
    point_cols = dominant_feature_colours(z, alive_idx, feat_cols)
    if labels is not None:
        label_cols = {lbl: (0, 0, 0, 0.2) for lbl in set(labels)}
    else:
        label_cols = None
    # Plot activations
    plot_activations(
        activations,
        ax=ax,
        labels=labels,
        colours=point_cols,
        label_cols=label_cols,
        short_labels=short_labels,
    )
    # Plot atoms
    if isinstance(model, SparseLinearAE):
        plot_atom_vectors(
            model,
            alive_idx,
            feat_cols,
            z=z,
            ax=ax,
        )
    elif isinstance(model, SparseCircularAE):
        plot_atom_loops(
            model,
            alive_idx,
            feat_cols,
            z=z,
            ax=ax,
        )
    # Annotate evaluation metrics
    annotate_eval(ax, evaluation)
    return ax


def plot_recovered_angles_polar(
    angles, labels, *, feat_idx=0, ax=None, short_labels=None
):
    """Polar plot: each point placed on a circle at its SCAE-recovered
    angle, coloured by label. The circular mean of each label is marked
    by a radial spoke labelled with the concept value.

    If the SCAE has correctly recovered the concept's circular structure,
    labels should appear at distinct angular positions, ideally in the
    natural semantic order (Mon→Sun / Jan→Dec).

    Args:
        angles: (N,) array of recovered angles in radians.
        feat_idx: Index of the feature to plot (if multiple features).
        labels:           (N,) array of string labels.
        ax: Matplotlib axis. If None, a new figure and axis are created.
        short_labels: optional dict mapping label to short display form.
    """
    if labels is None:
        labels = np.array([""] * len(angles))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

    angles = np.asarray(angles)[:, feat_idx]
    rng = np.random.default_rng(0)
    label_cols = label_colours(labels)
    unique = list(dict.fromkeys(labels))  # preserve order
    # Individual points at (angle, radius=1) with jittered radius
    for lbl in unique:
        mask = labels == lbl
        ang = angles[mask]
        r = 1.0 + 0.02 * rng.standard_normal(ang.shape)  # jitter radius
        ax.scatter(
            x=ang,
            y=r,
            s=30,
            alpha=0.5,
            color=label_cols[lbl],
            edgecolors="white",
            linewidths=0.3,
        )
        # Circular mean spoke + label
        mean = float(
            np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)
        )
        ax.plot(
            [mean, mean],
            [0, 1.2],
            color=label_cols[lbl],
            lw=2,
            alpha=0.8,
        )
        ax.text(
            mean,
            1.25,
            get_short_label(short_labels, lbl),
            color=label_cols[lbl],
            fontweight="bold",
            fontsize=14 if len(unique) > 10 else 18,
        )

    ax.set_ylim(0, 1.5)
    ax.set_yticklabels([])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
    ax.grid(alpha=0.3)
    return ax


def plot_recovered_angles_strip(
    angles: np.ndarray,
    labels: np.ndarray,
    short_labels=None,
    feat_idx: int = 0,
    ax=None,
):
    """Strip plot of recovered angles per label.

    More precise than the polar plot: shows within-label spread and
    cross-label ordering. A well-separated, monotonic pattern indicates
    the SCAE has recovered meaningful circular structure.

    Args:
        angles: (N,) array of recovered angles in radians.
        feat_idx: Index of the feature to plot (if multiple features).
        labels:           (N,) array of string labels.
        ax: Matplotlib axis. If None, a new figure and axis are created.
    """
    if labels is None:
        labels = np.array([""] * len(angles))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

    angles = np.asarray(angles)[:, feat_idx] % (2 * np.pi)
    rng = np.random.default_rng(0)
    label_cols = label_colours(labels)
    unique = list(dict.fromkeys(labels))  # preserve order

    for i, lbl in enumerate(unique):
        mask = labels == lbl
        ang = angles[mask]
        xs = i + 0.08 * rng.standard_normal(mask.sum())
        ax.scatter(
            xs,
            ang,
            s=35,
            alpha=0.6,
            color=label_cols[lbl],
            edgecolors="white",
            linewidths=0.4,
        )
        mean = float(
            np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)
        )
        ax.plot(
            [i - 0.32, i + 0.32],
            [mean, mean],
            color=label_cols[lbl],
            lw=2.5,
        )

    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(
        [get_short_label(short_labels, lb) for lb in unique],
        rotation=90,
        ha="center",
        fontsize=14 if len(unique) > 10 else 18,
    )
    ax.set_ylim(0, 2 * np.pi)
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels(
        [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"], fontsize=14
    )
    ax.grid(alpha=0.3, axis="y")
    return ax


def plot_sae_training(sae_history: dict, scae_history: dict, *, axs=None):
    """Plot training history of an SAE model.

    Args:
        sae_history: Dict with keys "loss", "mse", "l1", "sparsity".
        scae_history: Dict with keys "loss", "mse", "l1", "sparsity".
        axs: List of 3 Matplotlib axes. If None, a new figure and axes
            are created.
    """
    if axs is None:
        _, axs = plt.subplots(3, 1)

    for ax, metric in zip(axs, ["loss", "mse", "l1"]):
        epochs = range(1, len(sae_history[metric]) + 1)
        for history, label, col in zip(
            [sae_history, scae_history],
            ["SAE", "SCAE"],
            [C_SAE, C_SCAE],
        ):
            ax.plot(
                epochs,
                history[metric],
                label=label,
                color=col,
                lw=1.5,
            )
        ax.set_ylabel(metric.upper())
        ax.set_yscale("log")
    axs[-1].set_xlabel("Epoch")

    return axs
