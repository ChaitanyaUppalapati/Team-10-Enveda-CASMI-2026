"""Load several models' CandidateScores and align them on (query_id, inchikey14)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from casmi.interfaces import INCHIKEY14, QUERY_ID, SCORE, read_scores


def align_scores(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join per-model score tables into one wide table.

    Returns columns ``query_id, inchikey14`` plus one column per model id. A model that did
    not score a candidate leaves NaN; that is information (the candidate was outside its
    list), so it is kept until normalisation.
    """
    if not frames:
        raise ValueError("no input models given")
    wide: pd.DataFrame | None = None
    for model_id, df in frames.items():
        part = df[[QUERY_ID, INCHIKEY14, SCORE]].rename(columns={SCORE: model_id})
        wide = part if wide is None else wide.merge(part, on=[QUERY_ID, INCHIKEY14], how="outer")
    return wide.sort_values([QUERY_ID, INCHIKEY14]).reset_index(drop=True)


def load_inputs(model_ids: Sequence[str], split: str, root=None) -> pd.DataFrame:
    """Read ``outputs/scores/<model_id>/<split>.parquet`` for every id and align them."""
    return align_scores({m: read_scores(m, split, root) for m in model_ids})
