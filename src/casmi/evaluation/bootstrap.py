"""Bootstrap confidence intervals for MRR@25 (final model selection). Owner: Member 5."""

from __future__ import annotations

import pandas as pd


def bootstrap_ci(
    reciprocal_ranks: pd.Series, n: int = 1000, alpha: float = 0.05, seed: int = 2026
) -> tuple[float, float, float]:
    """Return (mean, low, high) by resampling queries."""
    raise NotImplementedError("TODO(Member 5): bootstrap CI")
