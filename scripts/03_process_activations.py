"""Process LLM activations for analysis."""

import os
import argparse
import glob
import numpy as np

from src.data.preprocess import preprocess_activations
from src.data.load import load_activations

from scripts.config import (
    DEFAULT_EXTRACT_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_PCA_COMPONENTS,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process LLM activations.")
    parser.add_argument(
        "--extract-dir",
        default=DEFAULT_EXTRACT_DIR,
        help="Directory containing extracted activations.",
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Directory to save processed activations.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help="Number of PCA components to retain during preprocessing.",
    )
    return parser.parse_args()


def main():
    """Process LLM activations based on command-line arguments."""
    args = parse_args()
    print(f"Extracted activations directory: {args.extract_dir}")
    print(f"Processed activations will be saved to: {args.processed_dir}")
    print(f"PCA components to retain: {args.pca_components}")
    # Create processed directory if it doesn't exist
    os.makedirs(args.processed_dir, exist_ok=True)
    # Get all .npz files in the extract directory
    activation_files = glob.glob(os.path.join(args.extract_dir, "*.npz"))
    # Process each activation file
    for activation_file in activation_files:
        print(f"Processing {activation_file}...")
        data = load_activations(activation_file)
        # Preprocess the activations
        processed_activations = preprocess_activations(
            data["activations"],
            template_ids=data.get("template_ids"),
            pca_components=args.pca_components,
        )
        # Save the processed activations
        data["activations"] = processed_activations
        np.savez(
            os.path.join(
                args.processed_dir, os.path.basename(activation_file)
            ),
            **data,
        )


if __name__ == "__main__":
    main()
