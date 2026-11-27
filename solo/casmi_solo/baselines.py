"""Candidate pool, mass window lookup, and the two baselines.

- B1 (mass only): candidates within ``PPM`` of the query's neutral mass, most-measured structures first.
- B2 (library search): candidates that have library spectra are ranked by the best cosine between any
  query spectrum and any of their library spectra (same ionisation mode); the rest follow in B1 order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import INTERIM, SpectrumReader, load_meta
from .spectra import table_to_csr, widen

PPM = 10.0


@dataclass
class Pool:
    """Candidate structures sorted by mass, one row per official key."""

    table: pd.DataFrame  # key, smiles, mass, prior, source

    @classmethod
    def build(cls, frames: list[pd.DataFrame]) -> Pool:
        t = pd.concat(frames, ignore_index=True)
        t = t.dropna(subset=["key", "mass"]).sort_values("prior", ascending=False).drop_duplicates("key")
        t = t.sort_values("mass").reset_index(drop=True)
        return cls(t)

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> Pool:
        """Already merged one-row-per-key table (``candidates.build_pool_table``)."""
        return cls(table.dropna(subset=["key", "mass"]).sort_values("mass").reset_index(drop=True))

    def window(self, mass: float, ppm: float = PPM) -> np.ndarray:
        m = self.table.mass.to_numpy()
        tol = mass * ppm * 1e-6
        return np.arange(np.searchsorted(m, mass - tol), np.searchsorted(m, mass + tol, side="right"))


def train_pool_frame(structures: pd.DataFrame, exclude_keys=()) -> pd.DataFrame:
    s = structures[~structures.inchikey14.isin(set(exclude_keys))]
    return pd.DataFrame({"key": s.key, "smiles": s.smiles, "mass": s.mass,
                         "prior": np.log1p(s.n_spectra), "source": "train"})


def molecule_mass(spectra: pd.DataFrame) -> pd.Series:
    """Neutral mass per molecule: median over its spectra (robust to one odd adduct)."""
    return spectra.groupby("molecule_id").neutral_mass.median()


def rank_b1(pool: Pool, mass: float, ppm: float = PPM) -> list[int]:
    idx = pool.window(mass, ppm)
    if len(idx) == 0:
        return []
    t = pool.table
    err = np.abs(t.mass.to_numpy()[idx] - mass)
    order = np.lexsort((err, -t.prior.to_numpy()[idx]))
    return idx[order].tolist()


# ---- spectral library ----

class Library:
    """Binned spectra of every good-label training row, with row -> official key and ionisation mode."""

    def __init__(self, matrix: sp.csr_matrix, row_id: np.ndarray, key: np.ndarray, mode: np.ndarray):
        self.matrix, self.row_id, self.key, self.mode = matrix, row_id, key, mode
        order = np.argsort(key, kind="stable")
        self._order, self._sorted = order, key[order]

    @classmethod
    def load_or_build(cls, structures: pd.DataFrame, refresh: bool = False) -> Library:
        path = INTERIM / "library_v1.npz"
        if path.exists() and not refresh:
            z = np.load(path, allow_pickle=True)
            m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
            return cls(m, z["row_id"], z["key"], z["mode"])
        meta = load_meta()
        good = meta.good_label.to_numpy()
        key_of = dict(zip(structures.inchikey14, structures.key, strict=True))
        reader, blocks, rows = SpectrumReader(), [], []
        t0 = time.time()
        for start, tab in reader.iter_groups(["ms2_mzs", "ms2_normalized_intensities"]):
            n = tab.num_rows
            local = np.flatnonzero(good[start:start + n])
            tab = tab.take(local)
            blocks.append(table_to_csr(tab, meta.precursor_mz.to_numpy()[start + local]))
            rows.append(start + local)
            print(f"  library: {start + n:,} rows read ({time.time() - t0:.0f}s)", flush=True)
        m = sp.vstack(blocks).tocsr()
        row_id = np.concatenate(rows)
        key = meta.inchikey14.to_numpy()[row_id]
        key = np.array([key_of[k] for k in key], dtype=object)
        mode = (meta.ionization_mode.to_numpy()[row_id] == "positive").astype(np.int8)
        np.savez(path, data=m.data, indices=m.indices, indptr=m.indptr, shape=np.array(m.shape),
                 row_id=row_id, key=key, mode=mode)
        return cls(m, row_id, key, mode)

    def rows_for_keys(self, keys, allowed: np.ndarray | None = None, mode: int | None = None) -> np.ndarray:
        keys = np.asarray(keys, dtype=object)
        lo = np.searchsorted(self._sorted, keys, side="left")
        hi = np.searchsorted(self._sorted, keys, side="right")
        if len(keys) == 0 or (hi - lo).sum() == 0:
            return np.zeros(0, dtype=np.int64)
        idx = np.concatenate([self._order[a:b] for a, b in zip(lo, hi, strict=True) if b > a])
        if allowed is not None:
            idx = idx[allowed[idx]]
        if mode is not None:
            idx = idx[self.mode[idx] == mode]
        return idx


def rank_b2(pool: Pool, lib: Library, mass: float, queries: sp.csr_matrix, modes: np.ndarray,
            allowed: np.ndarray | None = None, ppm: float = PPM) -> tuple[list[int], np.ndarray]:
    """Returns (pool indices ranked, best cosine per ranked candidate; -1 when the candidate has no spectra)."""
    b1 = np.array(rank_b1(pool, mass, ppm), dtype=np.int64)
    if len(b1) == 0:
        return [], np.zeros(0)
    best = b2_scores(pool, lib, b1, queries, modes, allowed)
    order = np.argsort(-best, kind="stable")  # stable: ties (incl. -1) keep B1 order
    return b1[order].tolist(), best[order]


def b2_scores(pool: Pool, lib: Library, idx: np.ndarray, queries: sp.csr_matrix, modes: np.ndarray,
              allowed: np.ndarray | None = None) -> np.ndarray:
    """Best cosine between any query spectrum and any library spectrum of each pool candidate ``idx``
    (same ionisation mode). -1 for candidates without library spectra."""
    cand_keys = pool.table.key.to_numpy()[idx]
    best = np.full(len(idx), -1.0)
    pos = {k: i for i, k in enumerate(cand_keys)}
    qw = widen(queries)
    for mode in np.unique(modes):
        rows = lib.rows_for_keys(cand_keys, allowed, int(mode))
        if len(rows) == 0:
            continue
        sims = (lib.matrix[rows] @ qw[np.flatnonzero(modes == mode)].T).toarray().max(axis=1)
        s = pd.Series(np.minimum(sims, 1.0)).groupby(lib.key[rows]).max()
        for k, v in s.items():
            i = pos[k]
            best[i] = max(best[i], v)
    return best
