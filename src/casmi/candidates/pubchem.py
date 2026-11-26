"""Mass-filtered PubChem subset (ablation). Owner: Member 4.

Schedule task: "Add mass-filtered PubChem subset as candidates (ablation)".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pool import load_pool

PUBCHEM_FILE = "pubchem.parquet"


def load_pubchem_subset(candidates_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the mass-filtered PubChem candidate table (same columns as COCONUT)."""
    return load_pool(PUBCHEM_FILE, candidates_dir)
