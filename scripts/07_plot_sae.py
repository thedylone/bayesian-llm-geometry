"""Plot SAE results for LLM activations."""

import os
import argparse
import glob
import numpy as np

import matplotlib.pyplot as plt

from src.vis.plot_helpers import (
    configure_plotting,
    save_fig,
)
from src.vis.sae import (
    plot_fitted_sae,
    plot_sae_training,
    plot_recovered_angles_polar,
    plot_recovered_angles_strip,
)
from src.sae.models import SparseLinearAE, SparseCircularAE
from src.data.load import load_activations

from scripts.config import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_SAE_RESULTS_DIR,
    DEFAULT_FIGURES_DIR,
    DEFAULT_ALIVE_THRESHOLD,
    SHORT_LABELS,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot SAE results for LLM activations."
    )
    parser.add_argument(
        "--in-data",
        default=DEFAULT_PROCESSED_DIR,
        help="Input directory containing processed activations.",
    )
    parser.add_argument(
        "--in-results",
        default=DEFAULT_SAE_RESULTS_DIR,
        help="Input directory containing SAE results.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for all figures.",
    )
    parser.add_argument(
        "--alive-threshold",
        type=float,
        default=DEFAULT_ALIVE_THRESHOLD,
        help="Threshold for determining alive features.",
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help="Use poster-sized fonts and save to poster figures directory.",
    )
    return parser.parse_args()


def main():
    """Plot SAE results for LLM activations."""
    args = parse_args()
    print(f"Input data directory: {args.in_data}")
    print(f"Input results directory: {args.in_results}")
    print(f"Output figures directory: {args.out}")
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    configure_plotting(poster=args.poster)
    # Get list of data files
    data_files = glob.glob(os.path.join(args.in_data, "*.npz"))
    for data_file in data_files:
        data_filename = os.path.basename(data_file).replace(".npz", "")
        save_dir = os.path.join(args.out, "sae", data_filename)
        print(f"Processing {data_filename}...")
        data = load_activations(data_file)
        # Load SAE results
        result_file = os.path.join(args.in_results, f"{data_filename}_sae.npz")
        results = np.load(result_file, allow_pickle=True)
        sae = SparseLinearAE.load_checkpoint(results["sae_params"].item())
        scae = SparseCircularAE.load_checkpoint(results["scae_params"].item())

        # Plot fitted SAE
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plot_fitted_sae(
            data["activations"],
            ax=ax,
            labels=data.get("labels"),
            model=sae,
            evaluation=results["sae_eval"].item(),
            alive_threshold=args.alive_threshold,
            short_labels=SHORT_LABELS,
        )
        save_fig(fig, save_dir, "fitted_sae")
        # Plot fitted SCAE
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plot_fitted_sae(
            data["activations"],
            ax=ax,
            labels=data.get("labels"),
            model=scae,
            evaluation=results["scae_eval"].item(),
            alive_threshold=args.alive_threshold,
            short_labels=SHORT_LABELS,
        )
        save_fig(fig, save_dir, "fitted_scae")

        # Plot training curves
        fig, axs = plt.subplots(3, 1)
        plot_sae_training(
            results["sae_history"].item(),
            results["scae_history"].item(),
            axs=axs,
        )
        save_fig(fig, save_dir, "training_curves")

        # Plot recovered angles
        labels = data.get("labels")
        if labels is None:
            print("  No labels found in the data. Skipping angle plot.")
            continue
        angles = results["scae_eval"].item()["info"]["angles"]
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        active_k = np.argmax(
            results["scae_eval"].item()["info"]["feature_activity"]
        )
        plot_recovered_angles_polar(
            angles, labels, feat_idx=active_k, ax=ax, short_labels=SHORT_LABELS
        )
        save_fig(fig, save_dir, "angles_polar")
        fig, ax = plt.subplots()
        plot_recovered_angles_strip(
            angles, labels, feat_idx=active_k, ax=ax, short_labels=SHORT_LABELS
        )
        save_fig(fig, save_dir, "angles_strip")


if __name__ == "__main__":
    main()
