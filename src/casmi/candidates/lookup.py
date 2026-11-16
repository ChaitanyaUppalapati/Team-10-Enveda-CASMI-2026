"""Candidate retrieval by precursor mass / formula. Owner: Member 4."""

from __future__ import annotations

import pandas as pd

PROTON_MASS = 1.007276


def candidates_for_queries(
    spectra: pd.DataFrame, pool: pd.DataFrame, ppm: float = 10.0
) -> pd.DataFrame:
    """Return (query_id, inchikey14, smiles, ...) rows whose mass matches each precursor.

    Must account for the adduct when converting precursor m/z to neutral mass.
    """
    raise NotImplementedError("TODO(Member 4): mass/adduct-aware candidate lookup")
