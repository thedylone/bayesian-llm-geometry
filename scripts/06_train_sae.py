"""Train Sparse Autoencoders on LLM activations."""

import os
import argparse
import glob
import numpy as np

from src.sae.models import SparseLinearAE, SparseCircularAE
from src.data.load import load_activations

from scripts.config import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_SAE_RESULTS_DIR,
    DEFAULT_NUM_FEATURES,
    TRAINING_CONFIG,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoders on LLM activations."
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Input directory containing processed activations.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_SAE_RESULTS_DIR,
        help="Output directory for SAE results.",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=DEFAULT_NUM_FEATURES,
        help="Number of features for the Sparse Autoencoder.",
    )
    return parser.parse_args()


def main():
    """Train Sparse Autoencoders on LLM activations.

    This function loads processed activations from the specified input
    directory and trains Sparse Autoencoders on the data.
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
        result_file = os.path.join(args.out, f"{data_filename}_sae.npz")
        # Check if the result file already exists
        if os.path.exists(result_file):
            print(f"Results for {data_filename} already exist. Skipping.")
            continue
        print(f"Training SAE/SCAE on {data_filename}...")
        # Load activations
        data = load_activations(activation_file)
        input_dim = data["activations"].shape[1]
        # Train SAE
        sae = SparseLinearAE(
            input_dim=input_dim, num_features=args.num_features
        )
        sae_history = sae.train_model(
            X_train=data["activations"], config=TRAINING_CONFIG
        )
        # Evaluate SAE
        sae_eval = sae.evaluate_np(data["activations"])
        # Train SCAE
        scae = SparseCircularAE(
            input_dim=input_dim, num_features=args.num_features
        )
        scae_history = scae.train_model(
            X_train=data["activations"], config=TRAINING_CONFIG
        )
        # Evaluate SCAE
        scae_eval = scae.evaluate_np(data["activations"])
        # Save results
        sae_params = sae.model_params()
        scae_params = scae.model_params()
        np.savez(
            result_file,
            sae_params=sae_params,
            sae_history=sae_history,
            sae_eval=sae_eval,
            scae_params=scae_params,
            scae_history=scae_history,
            scae_eval=scae_eval,
        )


if __name__ == "__main__":
    main()
