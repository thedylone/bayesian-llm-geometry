"""
load.py — Load and preprocess LLM activations for BMC and SAE/SCAE.
"""

# pylint: disable=invalid-name

import numpy as np


def load_activations(
    path: str,
) -> dict:
    """Load saved activations and optionally filter to a subset of labels.

    Args:
        path:         Path to a .npz file.
    Returns:
        data_dict:    Dictionary containing the loaded data.
    """
    with np.load(path, allow_pickle=True) as data:
        data_dict = dict(data)
    return data_dict
