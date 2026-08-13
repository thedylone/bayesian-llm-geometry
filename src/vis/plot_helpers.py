"""
plot_helpers.py
"""

# pylint: disable=invalid-name
# Mathematical convention: uppercase for matrices (X, W, N), lowercase for
# vectors (w, z, mu). Disabling invalid-name globally for readability.


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator

C_DATA = "#B4233C"
C_LINEAR = "#3B82F6"
C_CIRCULAR = "#F59E0B"
C_SAE = "#8B5CF6"  # purple for linear SAE
C_SCAE = "#10B981"  # green for circular SCAE

MARKER_KWARGS = {
    "s": 30,
    "edgecolors": "white",
    "marker": "o",
}

# ── Configuration ───────────────────────────────────────────────────


def configure_plotting(poster=False):
    """Set matplotlib rcParams for publication-quality figures.

    Args:
        poster: If True, use larger fonts and thicker lines suitable
                for poster display.
    """
    params = {
        "figure.figsize": (3, 3),
        "figure.autolayout": True,
        "font.family": "serif",
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 12,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
    if poster:
        params = {
            **params,
            "figure.dpi": 150,
        }
    else:
        params = {
            **params,
            "figure.dpi": 200,
        }
    plt.rcParams.update(params)


def save_fig(fig, dirname, name):
    """Save figure as PDF and PNG."""
    os.makedirs(dirname, exist_ok=True)
    fig.savefig(f"{dirname}/{name}.pdf")
    fig.savefig(f"{dirname}/{name}.png")
    plt.close(fig)
    print(f"  Saved {name}.pdf/png")


# ── Geometry helpers ────────────────────────────────────────────────


def pca_project(X, n_components=2):
    """Project data onto its top principal components.

    Args:
        X: Data matrix, shape (n, d).
        n_components: Number of components to keep.

    Returns:
        Tuple of (projections, PC_axes, variance_ratios).
    """
    mu = X.mean(axis=0)
    X_centered = X - mu
    _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    proj = X_centered @ Vt[:n_components].T
    var_explained = (S[:n_components] ** 2) / (S**2).sum()
    return proj, Vt[:n_components], var_explained


# ── Plotting helpers ────────────────────────────────────────────────


def label_colours(labels: list[str]) -> dict[str, tuple]:
    """Map each label to a discrete color from tab10 / tab20."""
    unique = list(dict.fromkeys(labels))  # preserve order
    n = len(unique)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return {lbl: cmap(i % cmap.N) for i, lbl in enumerate(unique)}


def style_2d_axis(ax, X_ref):
    """Apply publication-style formatting to a 2D axis."""
    mins = X_ref.min(axis=0)
    maxs = X_ref.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_range = 0.5 * np.max(maxs - mins)
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # Hide tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def style_3d_axis(ax, X_ref, elev=20, azim=-45):
    """Apply publication-style formatting to a 3D axis.

    Args:
        ax: Matplotlib 3D axis.
        X_ref: Reference data (n, 3) used to set equal axis limits.
        elev: Elevation viewing angle in degrees.
        azim: Azimuth viewing angle in degrees.
    """
    mins = X_ref.min(axis=0)
    maxs = X_ref.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_range = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)

    ax.set_proj_type("persp", focal_length=0.2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))

    pane_colors = [
        # disable first two panes
        (1, 1, 1, 0.0),
        (1, 1, 1, 0.0),
        # (0.95, 0.97, 1.00, 0.16),
        # (0.97, 0.99, 0.95, 0.16),
        (0.99, 0.98, 0.95, 0.22),
    ]
    for axis, pane_color in zip((ax.xaxis, ax.yaxis, ax.zaxis), pane_colors):
        axis.pane.set_facecolor(pane_color)
        axis.line.set_color((1, 1, 1, 0))  # hide axis lines
        axis._axinfo["grid"].update(  # pylint: disable=protected-access
            {"linewidth": 0.25, "linestyle": "-", "color": "#CBD5E1"}
        )
        axis.set_major_locator(MaxNLocator(5))

    ax.xaxis.set_tick_params(color=(1, 1, 1, 0))
    ax.yaxis.set_tick_params(color=(1, 1, 1, 0))
    ax.zaxis.set_tick_params(color=(1, 1, 1, 0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def plot_plane_shadows(
    ax,
    points,
    color="black",
    min_alpha=0.02,
    max_alpha=0.18,
    height_gamma=1.0,
):
    """Project 3D points onto the three coordinate planes.

    Args:
        ax: Matplotlib 3D axis.
        points: Array of shape (n, 3).
        color: Shadow color.
        min_alpha: Opacity for highest points.
        max_alpha: Opacity for lowest points.
        height_gamma: Controls how quickly opacity fades with height.
    """
    z_floor = ax.get_zlim()[0]

    x_vals, y_vals, z_vals = points[:, 0], points[:, 1], points[:, 2]

    z_min, z_max = z_vals.min(), z_vals.max()
    z_span = max(z_max - z_min, 1e-12)
    z_norm = (z_vals - z_min) / z_span

    # Lower points cast darker shadows; higher points fade.
    shadow_alpha = max_alpha - (max_alpha - min_alpha) * (z_norm**height_gamma)
    base_rgba = np.array(mcolors.to_rgba(color))
    shadow_rgba = np.tile(base_rgba, (len(z_vals), 1))
    shadow_rgba[:, 3] = np.clip(shadow_alpha, 0.0, 1.0)

    kwargs = {
        **MARKER_KWARGS,
        "c": shadow_rgba,
        "edgecolors": "none",
        "zorder": 0,
    }

    ax.scatter3D(x_vals, y_vals, np.full_like(z_vals, z_floor), **kwargs)


def get_short_label(short_labels: dict | None, lbl: str) -> str:
    """Return a short display form of a label, or the original if missing."""
    if short_labels is None:
        return lbl
    return short_labels.get(lbl.lower(), lbl)


def add_labels_to_data(ax, X, labels, label_cols, *, short_labels=None):
    """Add text labels to 3D scatter points.

    Args:
        ax: Matplotlib axis.
        X: Data points, shape (n, d).
        labels: Names of discrete labels, shape (n,).
        label_cols: Dict mapping label to color.
    """
    unique_labels = list(dict.fromkeys(labels))  # preserve order
    ndim = 3 if ax.name == "3d" else 2
    for lbl in unique_labels:
        mask = labels == lbl
        # add label above the median point for this label
        get_lim = ax.get_zlim if ndim == 3 else ax.get_ylim
        median_point = np.median(X[mask], axis=0)[:ndim]
        median_point[-1] += 0.02 * (get_lim()[1] - get_lim()[0])
        ax.text(
            *median_point,
            get_short_label(short_labels, lbl),
            color=label_cols[lbl],
            ha="center",
            va="bottom",
        )
    return ax


def annotate_text(ax, text):
    """Add a Bayes factor annotation to the middle of an axis.

    Args:
        ax: Matplotlib axis.
        text: Text to display.
        y_offset: Vertical offset for the annotation.
    """
    kwargs = {
        "ha": "center",
        "bbox": {"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.7},
    }

    # For 3D axes, use text2D so annotation stays pinned to the panel corner.
    if hasattr(ax, "text2D"):
        ax.text2D(0.5, 0.98, text, transform=ax.transAxes, va="top", **kwargs)
    else:
        ax.text(0.5, 0, text, transform=ax.transAxes, va="bottom", **kwargs)


def plot_activations(
    X,
    *,
    ax=None,
    labels=None,
    colours=None,
    label_cols=None,
    short_labels=None,
):
    """Scatter data in 3D and colour points.

    Args:
        X: data array (n,3)
        ax: optional matplotlib axis (3D or 2D, default 3D)
        labels: optional array of labels for each point (n,)
        colours: optional array of colours for each point (n, 4)
        label_cols: optional dict mapping label to colour
        short_labels: optional dict mapping label to short display form

    Returns:
        ax: matplotlib axis
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    marker_kwargs = {**MARKER_KWARGS}
    if ax.name == "3d":
        style_3d_axis(ax, X)
        plot_plane_shadows(ax, X, color=(0.1, 0.1, 0.1))
        marker_kwargs["depthshade"] = True
        ndim = 3
    else:
        style_2d_axis(ax, X)
        ndim = 2

    ax.scatter(
        *X[:, :ndim].T,
        c=colours if colours is not None else C_DATA,
        **marker_kwargs,
    )
    if labels is not None:
        if label_cols is None:
            label_cols = label_colours(labels)
        add_labels_to_data(
            ax, X, labels, label_cols, short_labels=short_labels
        )

    return ax
