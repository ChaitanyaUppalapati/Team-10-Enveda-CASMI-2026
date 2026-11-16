"""Fixed-width binning of spectra for CNN / MLP inputs. Owner: Member 2."""

from __future__ import annotations

import numpy as np


def bin_spectrum(
    mzs: np.ndarray, intensities: np.ndarray, max_mz: float = 1000.0, bin_width: float = 0.1
) -> np.ndarray:
    """Return a dense vector of length ceil(max_mz / bin_width), max-pooled per bin."""
    raise NotImplementedError("TODO(Member 2): binning")
