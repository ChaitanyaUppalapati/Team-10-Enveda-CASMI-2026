"""Per-class MRR@25 (e.g. by compound class, adduct, library). Owner: Member 5."""

from __future__ import annotations

import pandas as pd


def per_class_mrr(reciprocal_ranks: pd.Series, classes: pd.Series) -> pd.DataFrame:
    """Return a table of class, n_queries, mrr."""
    raise NotImplementedError("TODO(Member 5): per-class MRR")
