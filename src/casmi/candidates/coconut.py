"""COCONUT natural-product snapshot.

Owner: Member 4. Schedule task: "Package COCONUT snapshot as Kaggle Dataset; candidate
coverage audit". Packaged via ``kaggle/datasets/coconut/``.
Output table columns: inchikey14, smiles, formula, monoisotopic_mass.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pool import load_pool

COCONUT_FILE = "coconut.parquet"


def load_coconut(candidates_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the COCONUT candidate table from ``candidates_dir``."""
    return load_pool(COCONUT_FILE, candidates_dir)
