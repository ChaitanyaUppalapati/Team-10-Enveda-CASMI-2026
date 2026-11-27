"""Adapter to the team protocol (Member 3's ``casmi_pipeline``). Owner: Member 4.

The team agreed to validate on Member 3's split (classes 1 and 2, balanced MRR@25), rank with the same
soft-Tanimoto routine and score with the same ``evaluate``, so numbers are comparable across members.
``casmi_pipeline`` is loaded by path (it lives in ``baselines/b1_mass_only`` today and moves to
``models/spectrum_encoders/common`` when Member 3's branch merges), or taken from ``sys.modules`` when a
Kaggle notebook has already written it out.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.utils.config import load_paths

from . import data as D
from . import features as F

ROOT = Path(__file__).resolve().parents[3]
PIPELINE_CANDIDATES = (
    ROOT / "baselines" / "b1_mass_only" / "casmi_pipeline.py",
    ROOT / "models" / "spectrum_encoders" / "common" / "casmi_pipeline.py",
)


def load_pipeline():
    """Import Member 3's ``casmi_pipeline`` (already-imported module, ``CASMI_PIPELINE`` path, or repo path)."""
    if "casmi_pipeline" in sys.modules:
        return sys.modules["casmi_pipeline"]
    paths = [Path(p) for p in [os.environ.get("CASMI_PIPELINE")] if p] + list(PIPELINE_CANDIDATES)
    for path in paths:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("casmi_pipeline", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["casmi_pipeline"] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"casmi_pipeline.py not found; looked in {[str(p) for p in paths]}")


@dataclass
class Protocol:
    """Everything the team protocol defines: split, candidate pool, fingerprints, test molecules."""

    cp: object
    data_dir: str
    meta: pd.DataFrame
    queries: pd.DataFrame
    lib_mask: np.ndarray
    truth: pd.DataFrame
    pool: pd.DataFrame
    pool_fp: np.ndarray  # packed bits, row-aligned with pool
    key_index: pd.Series
    test: pd.DataFrame
    test_path: str

    @property
    def train_path(self) -> str:
        return os.path.join(self.data_dir, "train.parquet")


def external_candidates(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Pool tables from ``casmi.candidates`` (inchikey14, smiles, formula, monoisotopic_mass) -> the
    ``key, smiles, mass`` frame that ``casmi_pipeline.build_pool`` accepts."""
    if not tables:
        return None
    ext = pd.concat(tables, ignore_index=True).drop_duplicates("inchikey14")
    return ext.rename(columns={"inchikey14": "key", "monoisotopic_mass": "mass"})[["key", "smiles", "mass"]]


def load_protocol(data_cfg: dict, external: list[pd.DataFrame] | None = None) -> Protocol:
    """Build split, pool (training structures + optional external candidates) and pool fingerprints."""
    cp = load_pipeline()
    explicit = data_cfg.get("data_dir")
    if not explicit and (Path(load_paths()["raw_dir"]) / "train.parquet").is_file():
        explicit = str(load_paths()["raw_dir"])
    data_dir = cp.find_data_dir(explicit)
    meta = cp.load_meta(data_dir)
    queries, lib_mask = cp.make_split(meta, max_query_spectra=data_cfg.get("max_query_spectra", 16))
    queries = cp.sample_molecules(queries, data_cfg.get("n_val", 3000))
    truth = cp.truth_table(queries).reset_index(drop=True)
    assert not lib_mask[queries.row_id.to_numpy()].any(), "validation spectra must not be in the library"
    pool = cp.build_pool(meta, lib_mask, external_candidates(external or []))
    bits = D.morgan_bits(pool.smiles, data_cfg.get("radius", 2), cp.FP_BITS)
    pool_fp = np.packbits(bits, axis=1)
    test, test_path = cp.load_test(data_dir)
    return Protocol(
        cp, data_dir, meta, queries, lib_mask, truth, pool, pool_fp,
        pd.Series(np.arange(len(pool)), index=pool.key.to_numpy()), test, test_path,
    )  # fmt: skip


def spectra_arrays(
    path: str, rows: pd.DataFrame, bin_width: float = F.BIN_WIDTH, losses: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Binned spectra ``(n, n_bins)`` and precursor/adduct/CE features for ``rows`` (any row order).

    ``rows`` needs ``row_id`` (position in the parquet file), ``precursor_mz`` and ``adduct``.
    """
    ids = rows["row_id"].to_numpy()
    order = np.argsort(ids, kind="stable")
    precursor = rows["precursor_mz"].to_numpy(dtype=float)
    X, ce = D.read_binned(path, ids[order], precursor[order], bin_width, losses)
    if not (order == np.arange(len(order))).all():  # restore the caller's row order (skipped when already sorted)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        X, ce = X[inverse], ce[inverse]
    return X, F.meta_features(precursor, rows["adduct"].astype(str).tolist(), ce)


def training_data(proto: Protocol, data_cfg: dict) -> dict:
    """Training spectra from the library rows (Member 3's ``select_train_rows``) plus structure targets."""
    rows = proto.cp.select_train_rows(proto.meta, proto.lib_mask, per_structure=data_cfg.get("per_structure", 4))
    rows = rows[rows.inchikey14.astype(str).isin(proto.key_index.index)].reset_index(drop=True)
    X, meta = spectra_arrays(proto.train_path, rows, **F.spec_args(data_cfg))
    y_idx = rows.inchikey14.astype(str).map(proto.key_index).to_numpy()
    return {"rows": rows, "X": X, "meta": meta, "y_idx": y_idx}


def formula_targets(pool: pd.DataFrame) -> np.ndarray:
    """Auxiliary target per pool structure: log element counts (zeros where the formula is unknown)."""
    return F.log_counts(np.stack([F.formula_counts(f) for f in pool.formula.fillna("")])).astype(np.float32)


def rankings_to_scores(rankings: dict, pool: pd.DataFrame, top_k: int = 25) -> pd.DataFrame:
    """Team rankings ``{molecule_id: [pool index, ...]}`` -> CandidateScores (score = 1 / rank)."""
    keys = pool.key.to_numpy()
    rows = [(mol, keys[i], 1.0 / (r + 1)) for mol, idx in rankings.items() for r, i in enumerate(idx[:top_k])]
    return pd.DataFrame(rows, columns=["query_id", "inchikey14", "score"])
