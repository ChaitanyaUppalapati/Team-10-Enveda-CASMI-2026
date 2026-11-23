"""Turn rows of spectra into per-molecule sets, and pad sets into batches. Owner: Member 7.

Training molecules can have hundreds of spectra across libraries, while a test molecule has
1-16 (median 3). To train pooling on the situation it will face, each training step draws a
random subset whose size follows that test-like distribution.
"""

from __future__ import annotations

import numpy as np
import torch

MAX_SET = 16  # hidden test: 1-16 spectra per molecule, median 3


def draw_set_sizes(rng: np.random.Generator, n: int, max_set: int = MAX_SET) -> np.ndarray:
    """Draw set sizes in [1, max_set] with median 3 (geometric, p=0.25, capped)."""
    return np.minimum(rng.geometric(0.25, size=n), max_set)


def group_indices(keys: np.ndarray) -> dict[str, np.ndarray]:
    """Map each molecule key to the row indices of its spectra (order preserved)."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    ends = np.r_[starts[1:], len(order)]
    return {sorted_keys[s]: order[s:e] for s, e in zip(starts, ends, strict=True)}


def sample_sets(groups: dict[str, np.ndarray], keys: list[str], rng: np.random.Generator,
                max_set: int = MAX_SET) -> list[np.ndarray]:
    """For each key, sample a test-like number of its spectra without replacement."""
    sizes = draw_set_sizes(rng, len(keys), max_set)
    out = []
    for key, size in zip(keys, sizes, strict=True):
        rows = groups[key]
        out.append(rng.choice(rows, size=min(int(size), len(rows)), replace=False))
    return out


def pad_sets(sets: list[np.ndarray], features: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather per-spectrum features into (B, N_max, D) with a (B, N_max) mask of real rows.

    Output stays on the device of ``features`` (CPU array or GPU tensor).
    """
    feats = torch.as_tensor(features)
    n_max = max(len(s) for s in sets)
    index = np.zeros((len(sets), n_max), dtype=np.int64)
    mask = np.zeros((len(sets), n_max), dtype=bool)
    for i, rows in enumerate(sets):
        index[i, : len(rows)] = rows
        mask[i, : len(rows)] = True
    index_t = torch.from_numpy(index).to(feats.device)
    mask_t = torch.from_numpy(mask).to(feats.device)
    out = feats[index_t] * mask_t.unsqueeze(-1).to(feats.dtype)
    return out, mask_t
