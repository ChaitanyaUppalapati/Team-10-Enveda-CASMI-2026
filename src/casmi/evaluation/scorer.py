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
    raise NotImplementedError("TODO(Member 5): MRR@25 with InChIKey14 matching")


def reciprocal_ranks(scores: pd.DataFrame, truth: pd.DataFrame, k: int = TOP_K) -> pd.Series:
    """Per-query reciprocal rank (index: query_id); used by bootstrap and per-class metrics."""
    raise NotImplementedError("TODO(Member 5): per-query reciprocal ranks")
