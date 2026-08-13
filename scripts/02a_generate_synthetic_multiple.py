"""Generate multiple synthetic datasets with different seeds."""

import os
import argparse
import concurrent.futures
from tqdm import tqdm

from src.data.synthetic import generate_for_seed

from scripts.config import (
    DEFAULT_N_SAMPLES,
    DEFAULT_DIM,
    DEFAULT_SIGMA2,
    DEFAULT_SYNTHETIC_DIR,
)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate multiple synthetic datasets."
    )
    parser.add_argument(
        "-n",
        "--num-datasets",
        type=int,
        default=50,
        help="Number of synthetic datasets to generate.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base random seed for reproducibility.",
    )
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
    """Run the synthetic data generation script multiple times."""
    args = parse_args()
    print(
        f"Generating {args.num_datasets} synthetic datasets"
        f"with {args.n_samples} samples, "
        f"dim={args.dim}, sigma^2={args.sigma2}."
    )
    print(f"Synthetic data will be saved to: {args.out}")
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(args.num_datasets):
            seed = args.base_seed + i
            future = executor.submit(
                generate_for_seed,
                seed,
                args.n_samples,
                args.dim,
                args.sigma2,
                args.out,
            )
            futures.append(future)
        with tqdm(total=len(futures), desc="Generating Datasets") as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()
