"""Generate synthetic data for experimentation."""

import os
import argparse

from src.data.synthetic import generate_for_seed

from scripts.config import (
    DEFAULT_N_SAMPLES,
    DEFAULT_DIM,
    DEFAULT_SEED,
    DEFAULT_SIGMA2,
    DEFAULT_SYNTHETIC_DIR,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic data.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help="Dimensionality of the synthetic data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        default=DEFAULT_SIGMA2,
        help="Variance of the noise added to the synthetic data.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_SYNTHETIC_DIR,
        help="Output directory for synthetic data.",
    )
    return parser.parse_args()


def main():
    """Generate synthetic data based on command-line arguments."""
    args = parse_args()
    print(
        f"Generating synthetic data with {args.n_samples} samples, "
        f"dim={args.dim}, seed={args.seed}, sigma^2={args.sigma2}."
    )
    print(f"Synthetic data will be saved to: {args.out}")
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    generate_for_seed(
        args.seed, args.n_samples, args.dim, args.sigma2, args.out
    )
    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()
