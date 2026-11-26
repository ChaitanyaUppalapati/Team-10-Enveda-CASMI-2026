"""Small local MRR@25 used for weight tuning (InChIKey14 match, rank by score).

Member 5 owns the official scorer (`casmi.evaluation.scorer`). This module exists so the
ensembles can tune without waiting for it; the two must agree, and tests/ checks the formula.
"""

from __future__ import annotations

import pandas as pd

from casmi.interfaces import INCHIKEY14, QUERY_ID, SCORE, TOP_K


def reciprocal_ranks(scores: pd.DataFrame, truth: pd.DataFrame, k: int = TOP_K) -> pd.Series:
    """Per-query reciprocal rank (index: query_id); 0 if the truth is not in the top ``k``.

    Ties are broken by ``inchikey14``, matching ``casmi.interfaces.rank_candidates``.
    """
    ranked = scores.sort_values([QUERY_ID, SCORE, INCHIKEY14], ascending=[True, False, True])
    ranked = ranked.assign(rank=ranked.groupby(QUERY_ID).cumcount() + 1)
    hit = ranked.merge(truth[[QUERY_ID, INCHIKEY14]], on=[QUERY_ID, INCHIKEY14], how="inner")
    hit = hit[hit["rank"] <= k]
    rr = pd.Series(0.0, index=pd.Index(truth[QUERY_ID].unique(), name=QUERY_ID))
    best = (1.0 / hit.groupby(QUERY_ID)["rank"].min())
    rr.loc[best.index] = best.to_numpy()
    return rr


def mrr_at_k(scores: pd.DataFrame, truth: pd.DataFrame, k: int = TOP_K) -> float:
    return float(reciprocal_ranks(scores, truth, k).mean())
