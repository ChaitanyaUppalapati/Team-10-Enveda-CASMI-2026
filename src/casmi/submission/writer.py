"""CandidateScores -> submission.csv.

Owner: Member 6. TODO(Member 6): confirm the exact submission columns from the
competition Data/Evaluation page and set ``SUBMISSION_COLUMNS``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from casmi.interfaces import TOP_K, rank_candidates

SUBMISSION_COLUMNS: tuple[str, ...] = ()  # TODO(Member 6): fill from competition docs


def to_submission(scores: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Rank, keep top-k per query and reshape to the competition format."""
    ranked = rank_candidates(scores, top_k=k)
    _ = ranked
    raise NotImplementedError("TODO(Member 6): reshape ranked candidates into submission format")


def write_submission(scores: pd.DataFrame, path: str | Path = "submission.csv") -> Path:
    """Write submission.csv after running the checker."""
    from casmi.submission.checker import check_submission

    sub = check_submission(to_submission(scores))
    path = Path(path)
    sub.to_csv(path, index=False)
    return path
