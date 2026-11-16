"""Load train/test parquet into the shared spectrum schema.

Owner: Member 2. Schedule task: "Load train/test parquet, confirm schema, share data access".
Output columns must follow ``casmi.interfaces.SPECTRUM_COLUMNS`` (+ ``TRAIN_LABEL_COLUMNS``
for train). If the real schema differs, update the contract in a PR to interfaces.py.
"""

from __future__ import annotations

import pandas as pd

from casmi.utils.config import load_paths


def load_train(columns: list[str] | None = None) -> pd.DataFrame:
    """Return training spectra with labels (smiles, inchikey14)."""
    paths = load_paths()
    _ = paths["raw_dir"]
    raise NotImplementedError("TODO(Member 2): read train parquet from raw_dir and map columns to SPECTRUM_COLUMNS")


def load_test(columns: list[str] | None = None) -> pd.DataFrame:
    """Return test spectra (no labels)."""
    raise NotImplementedError("TODO(Member 2): read test parquet from raw_dir and map columns to SPECTRUM_COLUMNS")
