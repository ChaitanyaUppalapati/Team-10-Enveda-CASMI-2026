"""Fixed-width maximum-intensity binning. Owner: Member 2."""

from __future__ import annotations

import numpy as np


def bin_spectrum(
    mzs: np.ndarray,
    intensities: np.ndarray,
    max_mz: float = 1000.0,
    bin_width: float = 0.1,
    *,
    sqrt_intensity: bool = False,
) -> np.ndarray:
    """Filter invalid peaks, optionally sqrt once, then maximum-pool per bin.

    Defaults preserve the public function's existing 1000/0.1 grid. The CNN
    explicitly requests 2200/1.0 and sqrt_intensity=True.
    """
    if not np.isfinite(max_mz) or not np.isfinite(bin_width) or max_mz <= 0 or bin_width <= 0:
        raise ValueError("max_mz and bin_width must be finite and positive")
    mzs, intensities = np.asarray(mzs, dtype=np.float64), np.asarray(intensities, dtype=np.float32)
    if mzs.ndim != 1 or intensities.ndim != 1 or mzs.shape != intensities.shape:
        raise ValueError("Spectrum peak arrays must be one-dimensional and equally sized")
    valid = np.isfinite(mzs) & np.isfinite(intensities) & (mzs >= 0) & (mzs < max_mz) & (intensities >= 0)
    vector = np.zeros(int(np.ceil(max_mz / bin_width)), dtype=np.float32)
    values = np.sqrt(intensities[valid]) if sqrt_intensity else intensities[valid]
    np.maximum.at(vector, np.floor(mzs[valid] / bin_width).astype(np.int64), values)
    return vector
