"""Run Bayesian model comparison on LLM activations."""

import os
import argparse
import glob
import numpy as np

from src.bmc.models import LinearModel, CircularModel
from src.data.load import load_activations

from scripts.config import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_BMC_RESULTS_DIR,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian model comparison on LLM activations."
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Input directory containing processed activations.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_BMC_RESULTS_DIR,
        help="Output directory for BMC results.",
    )
    return parser.parse_args()


def main():
    """Run Bayesian model comparison on LLM activations.

    This function loads processed activations from the specified input
    directory, fits both linear and circular generative models to the data,
    computes the BF (Bayes Factor) for model comparison, and saves the results
    to the specified output directory.
    """
    args = parse_args()
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.out}")
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    # Find all processed activation files in the input directory
    activation_files = glob.glob(os.path.join(args.input_dir, "*.npz"))
    if not activation_files:
        print(f"No activation files found in {args.input_dir}. Exiting.")
        return

    for activation_file in activation_files:
        data_filename = os.path.basename(activation_file).replace(".npz", "")
        result_file = os.path.join(args.out, f"{data_filename}_bmc.npz")
        # Check if the result file already exists
        if os.path.exists(result_file):
            print(f"Results for {data_filename} already exist. Skipping.")
            continue
        print(f"Running BMC on {data_filename}...")
        # Load activations
        data = load_activations(activation_file)
        # Fit linear model
        linear_model = LinearModel()
        linear_params = linear_model.fit(X=data["activations"])
        # Fit circular model
        circular_model = CircularModel()
        circular_params = circular_model.fit(X=data["activations"])
        # Compute Bayes Factor
        log_bf = linear_model.log_evidence - circular_model.log_evidence
        print(f"  Log Bayes Factor (Linear vs Circular): {log_bf:.4f}")
        # Save results
        np.savez(
            result_file,
            log_bf=np.asarray(log_bf),
            linear_mu=np.asarray(linear_params["mu"]),
            linear_w=np.asarray(linear_params["w"]),
            linear_sigma2=np.asarray(linear_params["sigma2"]),
            linear_log_mls=np.asarray(linear_params["log_mls"]),
            circular_mu=np.asarray(circular_params["mu"]),
            circular_W=np.asarray(circular_params["W"]),
            circular_sigma2=np.asarray(circular_params["sigma2"]),
            circular_log_mls=np.asarray(circular_params["log_mls"]),
        )


if __name__ == "__main__":
    main()
