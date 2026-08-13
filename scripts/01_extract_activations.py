"""CLI for LLM activation extraction."""

import argparse
import torch

from src.data.extract import (
    build_prompts_for_concept,
    extract_concept_activations,
)

from scripts.config import (
    CONCEPTS,
    DEFAULT_EXTRACT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCEPT,
    DEFAULT_DTYPE,
    DEFAULT_LAYER,
    DEFAULT_MODEL_NAME,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract LLM residual-stream activations."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--concept",
        default=DEFAULT_CONCEPT,
        choices=list(CONCEPTS.keys()),
        help="Which concept to extract activations for.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help=(
            "Residual-stream layer index (0 = embedding, 1..L = "
            "transformer layers)."
        ),
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_EXTRACT_DIR,
        help="Output directory for extracted activations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Forward-pass batch size (reduce if OOM).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=["float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def main():
    """Extract LLM activations for a given concept and layer."""
    args = parse_args()
    print(f"Model: {args.model}")
    print(f"Concept: {args.concept}")
    print(f"Layer: {args.layer}")
    print(f"Output path: {args.out}")
    # Create output directory if it doesn't exist
    args.out.mkdir(parents=True, exist_ok=True)
    # Get concept details and build prompts
    concept = CONCEPTS[args.concept]
    placeholder_map = {
        "days": "day",
        "months": "month",
        "sentiment": "word",
        "size": "size",
    }
    placeholder = placeholder_map.get(args.concept, args.concept[:-1])
    prompts, target_words, template_ids = build_prompts_for_concept(
        concept["labels"],
        concept["templates"],
        placeholder,
    )
    out_path = args.out / f"{args.concept}_layer{args.layer}.npz"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    extract_concept_activations(
        model_name=args.model,
        prompts=prompts,
        target_words=target_words,
        template_ids=template_ids,
        layer=args.layer,
        out_path=out_path,
        device=args.device,
        batch_size=args.batch_size,
        dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()
