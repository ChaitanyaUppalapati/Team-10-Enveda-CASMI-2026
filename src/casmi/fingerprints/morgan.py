"""Morgan fingerprints for all training structures (shared targets for every model).

Owner: Member 1 (support: Member 4). Schedule task: "Morgan fingerprints for all training
structures". Output: ``processed_dir/fingerprints/morgan_r{radius}_{n_bits}.npy`` with a
row-aligned ``index.parquet`` (inchikey14). Requires ``pip install -e .[chem]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048


def morgan_fingerprint(
    smiles: str, radius: int = DEFAULT_RADIUS, n_bits: int = DEFAULT_N_BITS
) -> np.ndarray:
    """Return a uint8 bit vector for one SMILES string."""
    raise NotImplementedError("TODO(Member 1): RDKit Morgan fingerprint")


def compute_targets(
    structures: pd.DataFrame, radius: int = DEFAULT_RADIUS, n_bits: int = DEFAULT_N_BITS
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fingerprint every unique structure; return (matrix, index frame of inchikey14)."""
    raise NotImplementedError("TODO(Member 1): fingerprint all training structures and save to processed_dir")
