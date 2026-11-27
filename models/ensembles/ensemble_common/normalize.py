"""Per-query score normalisation so models with different score scales can be combined."""

from __future__ import annotations

import numpy as np
import pandas as pd

from casmi.interfaces import QUERY_ID

METHODS = ("rrf", "minmax", "zscore")
RRF_K = 60


def normalize(wide: pd.DataFrame, model_ids: list[str], method: str = "rrf") -> pd.DataFrame:
    """Return a table with the same keys and one normalised column per model.

    Each model is normalised *within each query*, over the candidates it scored.
    Candidates a model did not score get a neutral floor: 0 for ``rrf``/``minmax`` and
    (query min - 1) for ``zscore``, so "not in this model's list" never beats "scored low".

    - ``rrf``: reciprocal rank fusion term ``1 / (RRF_K + rank)``; scale-free, robust to outliers.
    - ``minmax``: (score - min) / (max - min) per query; 1.0 for a constant column.
    - ``zscore``: (score - mean) / std per query; 0.0 for a constant column.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    out = wide[[QUERY_ID, "inchikey14"]].copy()
    for m in model_ids:
        s = wide[m]
        g = s.groupby(wide[QUERY_ID])
        if method == "rrf":
            rank = g.rank(method="first", ascending=False)
            col = 1.0 / (RRF_K + rank)
            col = col.fillna(0.0)
        elif method == "minmax":
            lo, hi = g.transform("min"), g.transform("max")
            span = hi - lo
            col = ((s - lo) / span.where(span > 0, np.nan)).where(span > 0, 1.0)
            col = col.where(s.notna()).fillna(0.0)
        else:
            mean, std = g.transform("mean"), g.transform("std", ddof=0)
            z = ((s - mean) / std.where(std > 0, np.nan)).where(std > 0, 0.0).where(s.notna())
            floor = z.groupby(wide[QUERY_ID]).transform("min") - 1.0
            col = z.fillna(floor).fillna(-1.0)
        out[m] = col.astype(float)
    return out
