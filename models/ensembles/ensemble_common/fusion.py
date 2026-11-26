"""Late fusion: weighted sum of per-query-normalised model scores, with weight tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from casmi.interfaces import INCHIKEY14, QUERY_ID, SCORE, TOP_K

from .metrics import mrr_at_k


def fuse(norm: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Weighted sum of normalised columns -> CandidateScores (query_id, inchikey14, score)."""
    missing = [m for m in weights if m not in norm.columns]
    if missing:
        raise KeyError(f"no normalised column for: {missing}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    score = sum(norm[m] * (w / total) for m, w in weights.items())
    return pd.DataFrame({QUERY_ID: norm[QUERY_ID], INCHIKEY14: norm[INCHIKEY14], SCORE: score.astype(float)})


def tune_weights(
    norm: pd.DataFrame,
    truth: pd.DataFrame,
    model_ids: list[str],
    *,
    n_random: int = 200,
    seed: int = 2026,
    k: int = TOP_K,
) -> tuple[dict[str, float], float]:
    """Pick simplex weights that maximise MRR@k on ``truth``; returns (weights, best_mrr).

    Evaluates each model alone, the equal-weight mix, then ``n_random`` Dirichlet samples.
    Fit on the *development* split only; report on the frozen holdout.
    """
    rng = np.random.default_rng(seed)
    n = len(model_ids)
    cands = [np.eye(n)[i] for i in range(n)] + [np.full(n, 1.0 / n)]
    cands += list(rng.dirichlet(np.ones(n), size=n_random))
    best_w, best = cands[0], -1.0
    for w in cands:
        mrr = mrr_at_k(fuse(norm, dict(zip(model_ids, w, strict=True))), truth, k)
        if mrr > best + 1e-12:
            best_w, best = w, mrr
    return {m: float(x) for m, x in zip(model_ids, best_w, strict=True)}, float(best)
