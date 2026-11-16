"""Coverage audit: fraction of validation answers present in a candidate pool. Owner: Member 4."""

from __future__ import annotations

import pandas as pd


def coverage(candidates: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Share of queries whose true inchikey14 appears among their candidates."""
    raise NotImplementedError("TODO(Member 4): coverage audit")
