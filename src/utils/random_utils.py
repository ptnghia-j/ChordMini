from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed, include_python_random=False):
    """Seed torch / NumPy and optionally Python's ``random`` module."""
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if include_python_random:
        random.seed(seed)
    return seed


__all__ = ['set_random_seed']
