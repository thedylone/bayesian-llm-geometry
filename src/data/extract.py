"""
extract.py — Extract residual-stream activations for a concept.

For each target token (e.g. "Monday", "Tuesday", …) in a set of diverse
prompt templates, we run a forward pass through the model and collect the
residual-stream vector at the position of that token, at a chosen layer.

The result is a dict  {concept_value: np.ndarray of shape (N, d_model)}
that can be fed directly into models.compare_models() or the SAE/SCAE
training pipeline.

Requirements
------------
    pip install transformers accelerate torch
"""

# pylint: disable=invalid-name

import os
from pathlib import Path

import numpy as np
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = None

HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_token_positions(
    tokenizer, input_ids: torch.Tensor, target_word: str
) -> list[int]:
    """Return the index/indices of the first token(s) of target_word
    in input_ids.

    We tokenise the target word in isolation (with a leading space to match
    mid-sentence tokenisation) and search for that sub-sequence inside
    input_ids.

    Args:
        tokenizer: HF tokeniser.
        input_ids: 1-D integer tensor of the full prompt.
        target_word: The word whose position we want.

    Returns:
        List of token positions (one per token the word spans).
        Returns an empty list if the word is not found.
    """
    # Tokenise the target as a mid-sentence word (leading space matters for
    # BPE/SentencePiece tokenisers such as Llama).
    target_ids = tokenizer.encode(" " + target_word, add_special_tokens=False)

    ids_list = input_ids.tolist()
    n = len(target_ids)
    for i in range(len(ids_list) - n + 1):
        if ids_list[i : i + n] == target_ids:
            return list(range(i, i + n))

    # Fallback: try without leading space (handles sentence-initial position)
    target_ids_ns = tokenizer.encode(target_word, add_special_tokens=False)
    n = len(target_ids_ns)
    for i in range(len(ids_list) - n + 1):
        if ids_list[i : i + n] == target_ids_ns:
            return list(range(i, i + n))

    return []


@torch.no_grad()
def extract_residual_stream(
    model,
    tokenizer,
    prompts: list[str],
    target_words: list[str],
    template_ids: list[int] | None,
    layer: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> tuple[np.ndarray, list[str], list[int], list[int]]:
    """Extract residual-stream vectors at a given layer.

    For each (prompt, target_word) pair we:
      1. Tokenise the prompt.
      2. Find the token positions of target_word.
      3. Run a forward pass with output_hidden_states=True.
      4. Collect the hidden state at `layer` at the *last* token position
         of target_word (the most context-rich representation).

    Args:
        model:        HF CausalLM with output_hidden_states support.
        tokenizer:    Matching HF tokeniser.
        prompts:      List of prompt strings, length M.
        target_words: List of target words, one per prompt, length M.
        template_ids: Optional per-prompt template index. If given,
                      only the entries for successfully-extracted
                      prompts are kept in the returned `kept_template_ids`.
                      Pass None if no template tracking is needed.
        layer:        Which transformer layer (0 = embedding, 1..L = layers).
        batch_size:   Number of prompts to process simultaneously.
        device:       "cuda" or "cpu".

    Returns:
        activations:       np.ndarray of shape (M_found, d_model).
        kept_words:        List of target words for which a position was found.
        kept_template_ids: List of template indices aligned with kept_words
                           (empty list if template_ids was None).
        kept_indices:      List of original indices that were kept.
    """
    activations, kept_words, kept_indices = [], [], []
    kept_template_ids = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_words = target_words[start : start + batch_size]

        # Tokenise individually (prompts may have very different lengths).
        for idx_in_batch, (prompt, word) in enumerate(
            zip(batch_prompts, batch_words)
        ):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)
            input_ids = enc["input_ids"][0]  # (seq_len,)

            positions = get_token_positions(tokenizer, input_ids, word)
            if not positions:
                print(f"  [WARN] '{word}' not found in: {prompt!r}")
                continue

            target_pos = positions[-1]  # last sub-token of the word

            outputs = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[layer]  # (1, seq_len, d_model)
            vec = hidden[0, target_pos, :].cpu().float().numpy()
            activations.append(vec)
            kept_words.append(word)
            kept_indices.append(start + idx_in_batch)
            if template_ids is not None:
                kept_template_ids.append(template_ids[start + idx_in_batch])

        if (start // batch_size) % 5 == 0:
            print(
                f"  Processed {min(start + batch_size, len(prompts))}"
                f"/{len(prompts)} prompts…"
            )

    return np.array(activations), kept_words, kept_template_ids, kept_indices


# ── Main extraction pipeline ─────────────────────────────────────────────────


def build_prompts_for_concept(
    labels: list[str],
    templates: list[str],
    placeholder: str,
) -> tuple[list[str], list[str], list[int]]:
    """Build (prompt, target_word, template_idx) triples for a concept."""
    prompts, words, template_ids = [], [], []
    for label in labels:
        for t_idx, template in enumerate(templates):
            try:
                prompt = template.format(**{placeholder: label})
            except KeyError:
                prompt = template.replace("{" + placeholder + "}", label)
            prompts.append(prompt)
            words.append(label)
            template_ids.append(t_idx)

    return prompts, words, template_ids


def extract_concept_activations(
    model_name: str,
    prompts: list[str],
    target_words: list[str],
    template_ids: list[int] | None,
    layer: int,
    out_path: str,
    device: str = "cuda",
    batch_size: int = 4,
    dtype: torch.dtype = torch.float16,
):
    """Full pipeline: load model → build prompts → extract → save.

    The output .npz file contains:
        activations: (N, d_model) float32 array
        labels:      (N,) str array — the concept value for each row
        layer:       int scalar
        model_name:  str scalar

    Args:
        model_name:  HuggingFace model identifier.
        layer:       Residual-stream layer index (0 = embedding).
        out_path:    Output .npz path.
        device:      Torch device string.
        batch_size:  Forward-pass batch size.
        dtype:       Model dtype (float16 for GPU, float32 for CPU).
    """
    print(f"\n{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Layer   : {layer}")
    print(f"{'='*60}\n")

    if AutoTokenizer is None:
        raise ImportError(
            "The 'transformers' package is required for activation "
            "extraction. Install it with: pip install transformers "
            "accelerate"
        )

    # ── Load model & tokeniser ───────────────────────────────────────
    print("Loading tokeniser…")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model ({dtype})…")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        output_hidden_states=True,
    )
    model.eval()

    if device != "cuda":
        model = model.to(device)

    n_layers = model.config.num_hidden_layers
    print(
        f"  Model loaded. Layers: {n_layers}, d_model: "
        f"{model.config.hidden_size}"
    )
    if layer > n_layers:
        raise ValueError(
            f"Requested layer {layer} exceeds model depth {n_layers}."
        )

    # ── Extract activations ──────────────────────────────────────────
    print("\nExtracting activations…")
    acts, kept_words, kept_template_ids, _ = extract_residual_stream(
        model,
        tokenizer,
        prompts,
        target_words,
        template_ids,
        layer=layer,
        batch_size=batch_size,
        device=device,
    )
    print(
        f"\n  Extracted {len(acts)} activation vectors of dim {acts.shape[1]}."
    )

    # ── Save ─────────────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        activations=acts.astype(np.float32),
        labels=np.array(kept_words),
        template_ids=np.array(kept_template_ids, dtype=np.int32),
        layer=layer,
        model_name=model_name,
    )
    print(f"\n  Saved → {out_path}")
    return acts, kept_words
