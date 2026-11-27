"""Shared pieces of S3 training and inference: training-row selection, per-row features, batched prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .chem import TEST_ADDUCTS
from .fpmodel import batch_tensor, meta_features


def library_features(meta: pd.DataFrame, lib) -> tuple[np.ndarray, np.ndarray]:
    """(precursor, meta features) aligned with the library rows."""
    lm = meta.iloc[lib.row_id]
    precursor = lm.precursor_mz.to_numpy()
    feats = meta_features(precursor, lm.ionization_mode.to_numpy() == "positive", lm.adduct.astype(str).to_numpy(),
                          lm.ce_mean.to_numpy(), lm.instrument_type.astype(str).to_numpy() == "timsTOF")
    return precursor, feats


def query_features(qm: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    precursor = qm.precursor_mz.to_numpy()
    feats = meta_features(precursor, qm.ionization_mode.to_numpy() == "positive", qm.adduct.astype(str).to_numpy(),
                          qm.ce_mean.to_numpy(), qm.instrument_type.astype(str).to_numpy() == "timsTOF")
    return precursor, feats


def training_rows(meta, lib, banned_row, fp_keys, per_structure: int, seed: int):
    """Library indices for training, inner early-stopping rows, and the inner structure keys.

    Identical selection to the one used by ``scripts/train_fp.py`` for a given seed.
    """
    rng = np.random.default_rng(seed)
    lm = meta.iloc[lib.row_id]
    ok = (~banned_row[lib.row_id]) & lm.adduct.isin(TEST_ADDUCTS).to_numpy() & pd.Series(lib.key).isin(
        fp_keys).to_numpy()
    cand = pd.DataFrame({"i": np.flatnonzero(ok), "key": lib.key[ok]})
    cand = cand.sample(frac=1.0, random_state=seed).groupby("key").head(per_structure)
    keys = cand.key.unique()
    inner = set(rng.choice(keys, size=max(1, len(keys) // 100), replace=False))
    tr_rows = cand.i[~cand.key.isin(inner)].to_numpy()
    iv_rows = cand.i[cand.key.isin(inner)].to_numpy()[:20000]
    return tr_rows, iv_rows, inner, rng


@torch.no_grad()
def predict(model, X, rows, precursor, feats, device, batch=2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        out.append(model(batch_tensor(X, r, precursor[r], feats[r], device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2048), dtype=np.float32)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def key_fold(keys, n_folds: int) -> np.ndarray:
    """Deterministic fold of each structure key (stable across runs and machines)."""
    import hashlib

    return np.array([int(hashlib.md5(str(k).encode()).hexdigest()[:8], 16) % n_folds for k in keys])
