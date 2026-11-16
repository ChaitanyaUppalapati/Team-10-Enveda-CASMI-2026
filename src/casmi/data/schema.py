"""Schema checks for spectrum tables. Owner: Member 2."""

from __future__ import annotations

import pandas as pd

from casmi.interfaces import SPECTRUM_COLUMNS, SchemaError


def check_spectra(df: pd.DataFrame) -> pd.DataFrame:
    """Raise SchemaError if required spectrum columns are missing; return df."""
    missing = [c for c in SPECTRUM_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"spectrum table missing columns: {missing}")
    return df
