"""COCONUT natural-product snapshot.

Owner: Member 4. Schedule task: "Package COCONUT snapshot as Kaggle Dataset; candidate
coverage audit". Packaged via ``kaggle/datasets/coconut/``.
Output table columns: inchikey14, smiles, formula, monoisotopic_mass.
"""

from __future__ import annotations

import pandas as pd


def load_coconut() -> pd.DataFrame:
    """Load the COCONUT candidate table from ``candidates_dir``."""
    raise NotImplementedError("TODO(Member 4): load COCONUT snapshot")
