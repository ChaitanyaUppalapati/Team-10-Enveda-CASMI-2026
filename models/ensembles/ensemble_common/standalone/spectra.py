"""Binned, L2-normalised spectra (built on Member 2's ``bin_spectrum``) and per-query pooling."""

from __future__ import annotations

import numpy as np

from casmi.preprocess import bin_spectrum


def bin_one(mzs, intensities, max_mz: float, bin_width: float) -> np.ndarray:
    """sqrt-intensity max-pooled bins, L2-normalised (zero vector stays zero)."""
    v = bin_spectrum(np.asarray(mzs), np.asarray(intensities), max_mz, bin_width, sqrt_intensity=True)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def bin_many(mzs_list, ints_list, max_mz: float, bin_width: float) -> np.ndarray:
    d = int(np.ceil(max_mz / bin_width))
    out = np.zeros((len(mzs_list), d), dtype=np.float32)
    for i, (m, s) in enumerate(zip(mzs_list, ints_list, strict=True)):
        out[i] = bin_one(m, s, max_mz, bin_width)
    return out


def pool(vecs: np.ndarray) -> np.ndarray:
    """Mean of a molecule's spectra, re-normalised: one vector per query."""
    v = np.asarray(vecs, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v
