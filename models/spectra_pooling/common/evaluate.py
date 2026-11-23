"""Candidate scoring and a stand-in MRR@25. Owner: Member 7.

PROTOTYPE: ``mrr_at_k`` is a stand-in until Member 5's ``casmi.evaluation.scorer`` exists.
It matches on inchikey14 exactly as the competition does after canonicalisation, because
candidates here are already identified by inchikey14.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from casmi.interfaces import INCHIKEY14, QUERY_ID, SCORE, TOP_K, validate_candidate_scores


def tanimoto_scores(pred: np.ndarray, cand_fps: np.ndarray) -> np.ndarray:
    """Continuous Tanimoto between one predicted probability vector and many 0/1 fingerprints."""
    cand = cand_fps.astype(np.float32)
    inter = cand @ pred.astype(np.float32)
    return inter / np.maximum(pred.sum() + cand.sum(1) - inter, 1e-8)


def score_queries(preds: np.ndarray, query_ids: list[str], candidates: dict[str, list[str]],
                  fp_lookup: dict[str, np.ndarray]) -> pd.DataFrame:
    """CandidateScores (query_id, inchikey14, score) for every query's candidate list."""
    frames = []
    for pred, qid in zip(preds, query_ids, strict=True):
        keys = candidates[qid]
        fps = np.stack([fp_lookup[k] for k in keys])
        frames.append(pd.DataFrame({QUERY_ID: qid, INCHIKEY14: keys, SCORE: tanimoto_scores(pred, fps)}))
    return validate_candidate_scores(pd.concat(frames, ignore_index=True))


def ranks_of_truth(scores: pd.DataFrame, truth: dict[str, str]) -> pd.Series:
    """1-based rank of the true inchikey14 per query (ties broken by key), NaN if absent."""
    s = scores.sort_values([QUERY_ID, SCORE, INCHIKEY14], ascending=[True, False, True]).copy()
    s["rank"] = s.groupby(QUERY_ID).cumcount() + 1
    hit = s[s[INCHIKEY14].to_numpy() == s[QUERY_ID].map(truth).to_numpy()]
    return hit.set_index(QUERY_ID)["rank"].reindex(list(truth)).astype(float)


def mrr_at_k(ranks: pd.Series, k: int = TOP_K) -> float:
    """Mean of 1/rank if rank <= k else 0 (missing counts as 0)."""
    r = ranks.to_numpy(dtype=float)
    return float(np.where(np.isfinite(r) & (r <= k), 1.0 / np.nan_to_num(r, nan=1.0), 0.0).mean())


def summarise(ranks: pd.Series) -> dict[str, float]:
    r = ranks.to_numpy(dtype=float)
    hit = lambda k: float(np.mean(np.isfinite(r) & (r <= k)))  # noqa: E731
    return {"mrr@25": mrr_at_k(ranks), "top1": hit(1), "top5": hit(5), "top25": hit(25), "n": len(r)}
