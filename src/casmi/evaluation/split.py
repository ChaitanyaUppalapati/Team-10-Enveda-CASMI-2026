"""10% structure-level validation split, grouped by inchikey14.

Owner: Member 5 (support: Member 2). The split is generated once, frozen and committed to
``data/splits/val_inchikey14.csv``. Everyone evaluates on it.
"""

from __future__ import annotations

import pandas as pd

VAL_FRACTION = 0.10
SPLIT_SEED = 2026
SPLIT_FILE = "val_inchikey14.csv"


def make_structure_split(
    train: pd.DataFrame, frac: float = VAL_FRACTION, seed: int = SPLIT_SEED
) -> pd.Series:
    """Return the inchikey14 values assigned to validation (no structure in both sides)."""
    raise NotImplementedError("TODO(Member 5): structure-level split")


def load_val_keys() -> set[str]:
    """Read the frozen validation inchikey14 set from ``splits_dir``."""
    raise NotImplementedError("TODO(Member 5): load frozen split")
