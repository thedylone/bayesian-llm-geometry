"""Run BMC on multiple synthetic datasets."""

import os
import argparse
import glob
import concurrent.futures
import pandas as pd
from tqdm import tqdm

from src.bmc.models import LinearModel, CircularModel
from src.data.load import load_activations

from scripts.config import (
    DEFAULT_PROCESSED_SYNTHETIC_DIR,
    DEFAULT_BMC_SYNTHETIC_RESULTS_DIR,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian model comparison on LLM activations."
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        default=DEFAULT_PROCESSED_SYNTHETIC_DIR,
        help="Input directory containing processed activations.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_BMC_SYNTHETIC_RESULTS_DIR,
        help="Output directory for BMC results.",
    )
    return parser.parse_args()


def process_single_file(activation_file):
    """Worker function to process a single activation file and compute BF."""
    data_filename = os.path.basename(activation_file).replace(".npz", "")

    # Get ground truth and dimension from filename
    file_parts = data_filename.split("_")
    # Load activations
    data = load_activations(activation_file)
    activations = data["activations"]
    # Fit linear model
    linear_model = LinearModel()
    linear_model.fit(X=activations)
    # Fit circular model
    circular_model = CircularModel()
    circular_model.fit(X=activations)
    # Compute Bayes Factor
    log_bf = linear_model.log_evidence - circular_model.log_evidence
    # Return a dictionary of the results
    return {
        "ground_truth": file_parts[0],
        "n_samples": activations.shape[0],
        "dim": int(file_parts[2].replace("d", "")),
        "sigma2": data["sigma2"].item(),
        "log_bf": log_bf,
    }


def process_activations_to_dataframe(activation_files):
    """Process activation files in parallel, compute log BF"""
    if not activation_files:
        print("No activation files found. Exiting.")
        return None

    log_bf_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_file, f) for f in activation_files
        ]
        with tqdm(
            total=len(futures), desc="Processing synthetic datasets"
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    log_bf_list.append(result)
                except (OSError, KeyError, ValueError) as e:
                    print(f"\nError processing a file: {e}")
                finally:
                    pbar.update(1)

    # Convert list to DataFrame and save to CSV
    return pd.DataFrame(log_bf_list)


def main():
    """Run Bayesian model comparison on synthetic datasets.

    Gets the log BF (Bayes Factor) for each synthetic dataset
    and outputs statistics on the log BF values across datasets.
    """
    args = parse_args()
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.out}")
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    # Check if results CSV already exists
    results_file = os.path.join(args.out, "bmc_results.csv")
    if os.path.exists(results_file):
        print(f"Results file found at {results_file}. Loading directly...")
        log_bf_df = pd.read_csv(results_file)
    else:
        # Find all processed activation files in the input directory
        activation_files = glob.glob(os.path.join(args.input_dir, "*.npz"))
        log_bf_df = process_activations_to_dataframe(activation_files)
        # If no files were found, exit
        if log_bf_df is None:
            return
        log_bf_df.to_csv(results_file, index=False)

    # Output summary statistics (for same ground truth, n_samples, dim, sigma2)
    print("\nSummary Statistics:")
    dataset_columns = ["ground_truth", "n_samples", "dim", "sigma2"]
    # Group by the dataset columns and calculate the mean of log_bf
    summary_df = (
        log_bf_df.groupby(dataset_columns)["log_bf"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    # Rename the columns for clarity
    summary_df = summary_df.rename(
        columns={
            "mean": "avg_log_bf",
            "sem": "err_log_bf",
            "count": "n_datasets",
        }
    )
    print(summary_df)


if __name__ == "__main__":
    main()
