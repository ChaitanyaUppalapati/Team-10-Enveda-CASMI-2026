"""Local MRR@25 scorer.

Owner: Member 5 (support: Member 6). Schedule task: "Local MRR@25 scorer (RDKit 2026.03.3,
InChIKey14 matching)". Must reproduce the leaderboard metric exactly: a query whose true
inchikey14 is not in the top 25 scores 0.
"""

from __future__ import annotations

import pandas as pd

from casmi.interfaces import TOP_K


def mrr_at_k(scores: pd.DataFrame, truth: pd.DataFrame, k: int = TOP_K) -> float:
    """Mean reciprocal rank over all queries in ``truth``.

    Args:
        scores: CandidateScores (query_id, inchikey14, score).
        truth: one row per query: (query_id, inchikey14).
    """
    return float(reciprocal_ranks(scores, truth, k).mean())


def reciprocal_ranks(scores: pd.DataFrame, truth: pd.DataFrame, k: int = TOP_K) -> pd.Series:
    """Per-query reciprocal rank (index: query_id); used by bootstrap and per-class metrics."""
    if k < 1:
        raise ValueError("k must be positive")
    if not truth["query_id"].is_unique:
        raise ValueError("truth must have one row per query")
    ranked = scores.sort_values("score", ascending=False, kind="stable").copy()
    ranked["rank"] = ranked.groupby("query_id", sort=False).cumcount() + 1
    matches = truth[["query_id", "inchikey14"]].merge(
        ranked.loc[ranked["rank"] <= k, ["query_id", "inchikey14", "rank"]],
        on=["query_id", "inchikey14"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    return (1.0 / matches.set_index("query_id")["rank"]).fillna(0.0)
