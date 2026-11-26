"""Coverage audit: fraction of validation answers present in a candidate pool. Owner: Member 4."""

from __future__ import annotations

import pandas as pd

from casmi.interfaces import INCHIKEY14, QUERY_ID


def per_query_hit(candidates: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """Boolean per truth query: is its true ``inchikey14`` among that query's candidates?

    ``truth`` has one row per query (``query_id``, ``inchikey14``). ``candidates`` needs
    ``query_id`` and ``inchikey14``. Queries with no candidates count as misses.
    """
    if truth[QUERY_ID].duplicated().any():
        raise ValueError("truth must have one row per query_id")
    found = candidates[[QUERY_ID, INCHIKEY14]].drop_duplicates()
    merged = truth[[QUERY_ID, INCHIKEY14]].merge(found, on=[QUERY_ID, INCHIKEY14], how="left", indicator=True)
    return pd.Series(merged["_merge"].eq("both").to_numpy(), index=truth[QUERY_ID].to_numpy(), name="hit")


def coverage(candidates: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Share of queries whose true inchikey14 appears among their candidates (0 if no queries)."""
    hits = per_query_hit(candidates, truth)
    return float(hits.mean()) if len(hits) else 0.0


def pool_membership(pool: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Share of truth structures present in the pool at all, ignoring the mass filter.

    The gap between this and :func:`coverage` is the loss caused by mass/adduct matching.
    """
    return float(truth[INCHIKEY14].isin(set(pool[INCHIKEY14])).mean()) if len(truth) else 0.0
