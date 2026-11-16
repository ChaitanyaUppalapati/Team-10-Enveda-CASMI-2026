"""Mass-filtered PubChem subset (ablation). Owner: Member 4.

Schedule task: "Add mass-filtered PubChem subset as candidates (ablation)".
"""

from __future__ import annotations

import pandas as pd


def load_pubchem_subset() -> pd.DataFrame:
    """Load the mass-filtered PubChem candidate table (same columns as COCONUT)."""
    raise NotImplementedError("TODO(Member 4): PubChem subset")
