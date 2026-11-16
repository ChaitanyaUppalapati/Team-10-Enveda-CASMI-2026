"""Turn predicted fingerprints into CandidateScores.

Owner: Member 1. Used by every fingerprint-predicting model (MLP, CNN, encoders,
compact models) so they all rank candidates the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tanimoto(pred: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """Continuous Tanimoto between one predicted vector and a candidate matrix."""
    raise NotImplementedError("TODO(Member 1): tanimoto similarity")


def rank_by_fingerprint(
    pred_fps: np.ndarray, query_ids: list[str], candidates: pd.DataFrame
) -> pd.DataFrame:
    """Score each query's candidate set; return CandidateScores (query_id, inchikey14, score)."""
    raise NotImplementedError("TODO(Member 1): candidate ranking by fingerprint similarity")
