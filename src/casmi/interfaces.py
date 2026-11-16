"""Shared contracts between every area of the project.

Owners: Member 1 + Member 5. Changes need approval from BOTH and must keep
``tests/test_interfaces.py`` passing; all eight members depend on this file.

Integration rule: models never import each other. Each model writes a
*CandidateScores* table with :func:`write_scores`; rerankers, ensembles, B3, the
scorer and the submission writer read it back with :func:`read_scores`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Metric constants
# --------------------------------------------------------------------------------------
TOP_K = 25
"""The leaderboard metric is MRR@25: only the top 25 candidates per query count."""

INCHIKEY14_RE = re.compile(r"^[A-Z]{14}$")

SPLITS = ("train", "val", "test")
"""Split names used in output paths. ``val`` is the frozen structure-level split."""

# --------------------------------------------------------------------------------------
# Column names
# --------------------------------------------------------------------------------------
QUERY_ID = "query_id"
"""The unit that gets one ranked list (and one reciprocal rank) on the leaderboard."""
SPECTRUM_ID = "spectrum_id"
"""A single MS/MS spectrum. Several spectra may belong to one query (see pooling models)."""
INCHIKEY14 = "inchikey14"
SCORE = "score"
RANK = "rank"

CANDIDATE_SCORE_COLUMNS = (QUERY_ID, INCHIKEY14, SCORE)

# Spectrum-level columns produced by casmi.data.load (Member 2 confirms against the
# competition parquet schema and updates this list in a contract PR).
SPECTRUM_COLUMNS = (
    SPECTRUM_ID,
    QUERY_ID,
    "mzs",  # list[float], peak m/z values
    "intensities",  # list[float], same length as mzs
    "precursor_mz",
    "adduct",
    "collision_energy",
)
TRAIN_LABEL_COLUMNS = ("smiles", INCHIKEY14)


class SchemaError(ValueError):
    """Raised when a table does not follow a shared contract."""


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def to_inchikey14(inchikey: str) -> str:
    """Return the first block (14 chars) of a full InChIKey; the metric's match key."""
    key = inchikey.strip().split("-")[0]
    if not INCHIKEY14_RE.match(key):
        raise SchemaError(f"not a valid InChIKey first block: {inchikey!r}")
    return key


@dataclass
class SpectrumRecord:
    """One MS/MS spectrum in memory (row form of the spectrum table)."""

    spectrum_id: str
    query_id: str
    mzs: list[float]
    intensities: list[float]
    precursor_mz: float
    adduct: str | None = None
    collision_energy: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.mzs) != len(self.intensities):
            raise SchemaError(
                f"spectrum {self.spectrum_id}: {len(self.mzs)} m/z values "
                f"vs {len(self.intensities)} intensities"
            )


def validate_candidate_scores(df: pd.DataFrame, *, max_per_query: int | None = None) -> pd.DataFrame:
    """Check that ``df`` is a valid CandidateScores table and return it unchanged.

    Rules: the required columns exist, there are no nulls, every ``inchikey14`` is 14
    uppercase letters, ``score`` is numeric and finite, and ``(query_id, inchikey14)``
    pairs are unique. ``max_per_query`` optionally caps the candidates per query
    (use ``TOP_K`` for final outputs).
    """
    missing = [c for c in CANDIDATE_SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"CandidateScores missing columns: {missing}")
    sub = df[list(CANDIDATE_SCORE_COLUMNS)]
    if sub.isna().any().any():
        raise SchemaError("CandidateScores contains nulls")
    keys = sub[INCHIKEY14].astype(str)
    bad = keys[~keys.str.fullmatch(INCHIKEY14_RE.pattern)]
    if len(bad):
        raise SchemaError(f"invalid inchikey14 values, e.g. {bad.iloc[0]!r}")
    if not pd.api.types.is_numeric_dtype(sub[SCORE]):
        raise SchemaError("score column must be numeric")
    if not np.isfinite(sub[SCORE].to_numpy(dtype=float)).all():
        raise SchemaError("score column must be finite")
    if sub.duplicated([QUERY_ID, INCHIKEY14]).any():
        raise SchemaError("duplicate (query_id, inchikey14) pairs")
    if max_per_query is not None:
        worst = sub.groupby(QUERY_ID).size().max()
        if worst > max_per_query:
            raise SchemaError(f"a query has {worst} candidates > max_per_query={max_per_query}")
    return df


def rank_candidates(df: pd.DataFrame, top_k: int | None = TOP_K) -> pd.DataFrame:
    """Sort by score (descending) within each query, add a 1-based ``rank``, keep top_k.

    Ties are broken by ``inchikey14`` so results are deterministic.
    """
    validate_candidate_scores(df)
    out = df.sort_values([QUERY_ID, SCORE, INCHIKEY14], ascending=[True, False, True]).copy()
    out[RANK] = out.groupby(QUERY_ID).cumcount() + 1
    if top_k is not None:
        out = out[out[RANK] <= top_k]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Score file convention: outputs/scores/<model_id>/<split>.parquet
# --------------------------------------------------------------------------------------
def scores_path(model_id: str, split: str, root: str | Path | None = None) -> Path:
    """Path where ``model_id`` stores its CandidateScores for ``split``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if not model_id or "/" in model_id or "\\" in model_id:
        raise ValueError("model_id must be non-empty with no path separators, e.g. 'family.arch'")
    if root is None:
        from casmi.utils.config import load_paths

        root = load_paths()["output_dir"]
    return Path(root) / "scores" / model_id / f"{split}.parquet"


def write_scores(
    df: pd.DataFrame, model_id: str, split: str, root: str | Path | None = None
) -> Path:
    """Validate and write a CandidateScores table; returns the file path."""
    validate_candidate_scores(df)
    path = scores_path(model_id, split, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[list(CANDIDATE_SCORE_COLUMNS)].to_parquet(path, index=False)
    return path


def read_scores(model_id: str, split: str, root: str | Path | None = None) -> pd.DataFrame:
    """Read and validate a CandidateScores table written by :func:`write_scores`."""
    return validate_candidate_scores(pd.read_parquet(scores_path(model_id, split, root)))


# --------------------------------------------------------------------------------------
# Model protocol: every models/*/arch*/ and baselines/*/ implements this shape
# --------------------------------------------------------------------------------------
@runtime_checkable
class ModelRunner(Protocol):
    """Minimal interface each model exposes to the integration scripts (B3, ensembles)."""

    model_id: str

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Train on spectrum rows (SPECTRUM_COLUMNS + TRAIN_LABEL_COLUMNS)."""
        ...

    def predict(self, spectra: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        """Score candidates for each query; return a CandidateScores table."""
        ...
