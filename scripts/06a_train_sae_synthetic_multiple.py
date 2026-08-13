"""Train SAE/SCAE on multiple synthetic datasets."""

import os
import argparse
import glob
import concurrent.futures
import queue
import pandas as pd
from tqdm import tqdm

from src.sae.models import SparseLinearAE, SparseCircularAE
from src.data.load import load_activations

from scripts.config import (
    DEFAULT_PROCESSED_SYNTHETIC_DIR,
    DEFAULT_SAE_SYNTHETIC_RESULTS_DIR,
    DEFAULT_NUM_FEATURES,
    TRAINING_CONFIG,
    DEFAULT_MAX_CONCURRENT_MODELS,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoders on LLM activations."
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        default=DEFAULT_PROCESSED_SYNTHETIC_DIR,
        help="Input directory containing processed activations.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_SAE_SYNTHETIC_RESULTS_DIR,
        help="Output directory for SAE results.",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=DEFAULT_NUM_FEATURES,
        help="Number of features for the Sparse Autoencoder.",
    )
    parser.add_argument(
        "--max_concurrent_models",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_MODELS,
        help="Maximum number of models to train on the GPU simultaneously.",
    )
    return parser.parse_args()


def process_single_file(activation_file, num_features, pbar_pos=1):
    """Worker function to process a single activation file and compute MSE."""
    data_filename = os.path.basename(activation_file).replace(".npz", "")

    # Check if results already exist for this file
    results_file = os.path.join(
        DEFAULT_SAE_SYNTHETIC_RESULTS_DIR, f"{data_filename}_results.csv"
    )
    if os.path.exists(results_file):
        return pd.read_csv(results_file).to_dict(orient="records")[0]
    # Get ground truth and dimension from filename
    file_parts = data_filename.split("_")
    # Load activations
    data = load_activations(activation_file)
    training_dict = {
        "X_train": data["activations"],
        "config": TRAINING_CONFIG,
        "pbar_pos": pbar_pos,
        "pbar_leave": False,
    }
    input_dim = data["activations"].shape[1]
    # Train SAE
    sae = SparseLinearAE(input_dim=input_dim, num_features=num_features)
    sae.train_model(**training_dict)
    # Evaluate SAE
    sae_eval = sae.evaluate_np(data["activations"])
    # Train SCAE
    scae = SparseCircularAE(input_dim=input_dim, num_features=num_features)
    scae.train_model(**training_dict)
    # Evaluate SCAE
    scae_eval = scae.evaluate_np(data["activations"])
    # Save results to CSV
    pd.DataFrame(
        [
            {
                "ground_truth": file_parts[0],
                "n_samples": data["activations"].shape[0],
                "dim": int(file_parts[2].replace("d", "")),
                "sigma2": data["sigma2"].item(),
                "sae_loss": sae_eval["loss"],
                "sae_mse": sae_eval["mse"],
                "sae_l1": sae_eval["l1"],
                "scae_loss": scae_eval["loss"],
                "scae_mse": scae_eval["mse"],
                "scae_l1": scae_eval["l1"],
            }
        ]
    ).to_csv(results_file, index=False)
    # Return a dictionary of the results
    return {
        "ground_truth": file_parts[0],
        "n_samples": data["activations"].shape[0],
        "dim": int(file_parts[2].replace("d", "")),
        "sigma2": data["sigma2"].item(),
        "sae_loss": sae_eval["loss"],
        "sae_mse": sae_eval["mse"],
        "sae_l1": sae_eval["l1"],
        "scae_loss": scae_eval["loss"],
        "scae_mse": scae_eval["mse"],
        "scae_l1": scae_eval["l1"],
    }


def process_single_file_with_queue(activation_file, num_features, pos_queue):
    """Worker function to process a single activation file and compute MSE."""
    worker_pos = pos_queue.get()
    try:
        return process_single_file(
            activation_file, num_features, pbar_pos=worker_pos
        )
    finally:
        pos_queue.put(worker_pos)  # Release the position back to the queue


def process_activations_to_dataframe(
    activation_files,
    num_features,
    *,
    max_concurrent_models=DEFAULT_MAX_CONCURRENT_MODELS,
):
    """Process activation files in parallel, compute log BF"""
    if not activation_files:
        print("No activation files found. Exiting.")
        return None

    eval_list = []
    pos_queue = queue.Queue()
    for i in range(1, max_concurrent_models + 1):
        pos_queue.put(i)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_concurrent_models
    ) as executor:
        futures = [
            # executor.submit(process_single_file, f, num_features)
            executor.submit(
                process_single_file_with_queue, f, num_features, pos_queue
            )
            for f in activation_files
        ]
        with tqdm(
            total=len(futures),
            desc="Training SAE/SCAEs",
            position=0,
            leave=True,
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    eval_list.append(result)
                except (OSError, KeyError, ValueError) as e:
                    print(f"\nError processing a file: {e}")
                finally:
                    pbar.update(1)

    # Convert list to DataFrame and save to CSV
    return pd.DataFrame(eval_list)


def main():
    """Train SAE and SCAE on synthetic datasets.

    Gets the loss, MSE, and L1 loss for each synthetic dataset
    and outputs statistics on each metric across datasets.
    """
    args = parse_args()
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.out}")
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    # Check if results CSV already exists
    results_file = os.path.join(args.out, "sae_results.csv")
    if os.path.exists(results_file):
        print(f"Results file found at {results_file}. Loading directly...")
        eval_df = pd.read_csv(results_file)
    else:
        # Find all processed activation files in the input directory
        activation_files = glob.glob(os.path.join(args.input_dir, "*.npz"))
        eval_df = process_activations_to_dataframe(
            activation_files,
            args.num_features,
            max_concurrent_models=args.max_concurrent_models,
        )
        # If no files were found, exit
        if eval_df is None:
            return
        eval_df.to_csv(results_file, index=False)

    # Output summary statistics (for same ground truth, n_samples, dim, sigma2)
    print("\nSummary Statistics:")
    dataset_columns = ["ground_truth", "n_samples", "dim", "sigma2"]
    # Group by the dataset columns and calculate the mean of loss, mse, and l1
    # for both SAE and SCAE
    summary_df = (
        eval_df.groupby(dataset_columns)[
            [
                "sae_loss",
                "sae_mse",
                "sae_l1",
                "scae_loss",
                "scae_mse",
                "scae_l1",
            ]
        ]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    print(summary_df)


if __name__ == "__main__":
    main()
