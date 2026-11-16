"""Reproducibility. Owner: Member 1."""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 2026


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed python, numpy and (if installed) torch. Returns the seed used."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
