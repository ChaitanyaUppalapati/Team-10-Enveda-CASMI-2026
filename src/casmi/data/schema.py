"""Spectrum table integrity checks. Owner: Member 2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from casmi.interfaces import SPECTRUM_COLUMNS, SchemaError


def check_spectra(df: pd.DataFrame) -> pd.DataFrame:
    """Check IDs, precursor values and peak arrays without changing the table."""
    if missing := set(SPECTRUM_COLUMNS) - set(df):
        raise SchemaError(f"spectrum table missing columns: {sorted(missing)}")
    for name in ("spectrum_id", "query_id"):
        if df[name].isna().any() or not df[name].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise SchemaError(f"{name} must contain nonempty string IDs")
    if df.spectrum_id.duplicated().any():
        raise SchemaError("Duplicate spectrum_id")
    precursor = pd.to_numeric(df.precursor_mz, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(precursor).all() or (precursor <= 0).any():
        raise SchemaError("precursor_mz must be finite and positive")
    for index, (mz, intensity) in enumerate(zip(df.mzs, df.intensities, strict=True)):
        try:
            mz, intensity = np.asarray(mz, dtype=float), np.asarray(intensity, dtype=float)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"Invalid peak arrays at row {index}") from exc
        if mz.ndim != 1 or intensity.ndim != 1 or mz.shape != intensity.shape:
            raise SchemaError(f"Unequal/non-vector peak arrays at row {index}")
        if not np.isfinite(mz).all() or not np.isfinite(intensity).all() or (mz < 0).any() or (intensity < 0).any():
            raise SchemaError(f"Non-finite/negative peaks at row {index}; clean peaks before validation")
    return df
