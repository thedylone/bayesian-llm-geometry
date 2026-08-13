"""Plot BMC results for LLM activations."""

import os
import argparse
import glob
import numpy as np

import matplotlib.pyplot as plt

from src.vis.plot_helpers import (
    configure_plotting,
    save_fig,
)
from src.vis.bmc import plot_bmc_scatter, plot_bmc_training
from src.data.load import load_activations

from scripts.config import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_BMC_RESULTS_DIR,
    DEFAULT_FIGURES_DIR,
    SHORT_LABELS,
)


def load_bmc_params(result_file):
    """Load BMC parameters from the explicit array schema."""
    data = np.load(result_file)
    return {
        "linear": {
            "mu": data["linear_mu"],
            "w": data["linear_w"],
            "sigma2": data["linear_sigma2"],
            "log_mls": data["linear_log_mls"],
        },
        "circular": {
            "mu": data["circular_mu"],
            "W": data["circular_W"],
            "sigma2": data["circular_sigma2"],
            "log_mls": data["circular_log_mls"],
        },
        "log_bf": data["log_bf"],
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot BMC results for LLM activations."
    )
    parser.add_argument(
        "--in-data",
        default=DEFAULT_PROCESSED_DIR,
        help="Input directory containing processed activations.",
    )
    parser.add_argument(
        "--in-results",
        default=DEFAULT_BMC_RESULTS_DIR,
        help="Input directory containing BMC results.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for all figures.",
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help="Use poster-sized fonts and save to poster figures directory.",
    )
    return parser.parse_args()


def main():
    """Plot BMC results for LLM activations."""
    args = parse_args()
    print(f"Input data directory: {args.in_data}")
    print(f"Input results directory: {args.in_results}")
    print(f"Output figures directory: {args.out}")
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    configure_plotting(poster=args.poster)
    # Get list of data files in the input data directory
    data_files = glob.glob(os.path.join(args.in_data, "*.npz"))
    for data_file in data_files:
        data_filename = os.path.basename(data_file).replace(".npz", "")
        save_dir = os.path.join(args.out, "bmc", data_filename)
        print(f"Processing {data_filename}...")
        # Load activations
        data = load_activations(data_file)
        # Load BMC results
        result_file = os.path.join(args.in_results, f"{data_filename}_bmc.npz")
        bmc_params = load_bmc_params(result_file)

        # Plot BMC with manifolds and BF in 3D
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plot_bmc_scatter(
            data["activations"],
            bmc_params,
            labels=data.get("labels"),
            ax=ax,
            short_labels=SHORT_LABELS,
        )
        save_fig(fig, save_dir, "fitted_bmc")

        # Plot BMC with manifolds and BF in 2D
        fig, ax = plt.subplots()
        plot_bmc_scatter(
            data["activations"],
            bmc_params,
            labels=data.get("labels"),
            ax=ax,
            short_labels=SHORT_LABELS,
        )
        save_fig(fig, save_dir, "fitted_bmc_2d")

        # Plot training log marginal likelihoods
        fig, ax = plt.subplots()
        plot_bmc_training(
            bmc_params["linear"]["log_mls"],
            bmc_params["circular"]["log_mls"],
            ax=ax,
        )
        save_fig(fig, save_dir, "training_log_mls")


if __name__ == "__main__":
    main()
