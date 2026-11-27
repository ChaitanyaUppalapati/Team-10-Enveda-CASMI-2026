"""Parquet access and fingerprints for the spectrum + formula models. Owner: Member 4.

Only the selected rows' peaks are read from parquet (row group by row group) and binned, so the 3 GB train
file is never held in memory. The split, candidate pool and training rows come from the team protocol
(``common/team.py``).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import features as F

PEAK_COLUMNS = ("ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev")


# ---------- peaks ----------
def read_binned(
    path: str | Path,
    positions: np.ndarray,
    precursor_mz: np.ndarray,
    bin_width: float = F.BIN_WIDTH,
    losses: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Binned spectra ``(n, n_bins)`` float16 and mean collision energy ``(n,)`` for sorted row ``positions``."""
    pf = pq.ParquetFile(path)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    positions = np.asarray(positions)
    X = np.zeros((len(positions), F.input_dim(bin_width, losses)), dtype=np.float16)
    ce = np.full(len(positions), np.nan)
    for g in range(pf.num_row_groups):
        lo, hi = np.searchsorted(positions, [starts[g], starts[g + 1]])
        if hi == lo:
            continue
        table = pf.read_row_group(g, columns=list(PEAK_COLUMNS)).take(positions[lo:hi] - starts[g])
        cols = {c: table.column(c).combine_chunks() for c in PEAK_COLUMNS}
        lens = pc.fill_null(pc.list_value_length(cols["ms2_mzs"]), 0).to_numpy()
        offsets = np.concatenate([[0], np.cumsum(lens)])
        X[lo:hi] = F.bin_flat_peaks(
            pc.list_flatten(cols["ms2_mzs"]).to_numpy(zero_copy_only=False),
            pc.list_flatten(cols["ms2_normalized_intensities"]).to_numpy(zero_copy_only=False),
            offsets,
            precursor_mz[lo:hi],
            bin_width=bin_width,
            losses=losses,
        ).astype(np.float16)
        ce[lo:hi] = _mean_ce(cols["collision_energy_ev"])
    return X, ce


def _mean_ce(col) -> np.ndarray:
    lens = pc.fill_null(pc.list_value_length(col), 0).to_numpy()
    values = pc.list_flatten(col).to_numpy(zero_copy_only=False)
    row = np.repeat(np.arange(len(lens)), lens)
    finite = np.isfinite(values)
    total = np.bincount(row[finite], weights=values[finite], minlength=len(lens))
    n = np.bincount(row[finite], minlength=len(lens))
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


# ---------- fingerprints ----------
def _morgan_chunk(args) -> np.ndarray:
    smiles, radius, n_bits = args
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is not None:
            out[i] = gen.GetFingerprintAsNumPy(mol)
    return out


def morgan_bits(smiles, radius: int = 2, n_bits: int = 2048, workers: int = 0, chunk: int = 5000) -> np.ndarray:
    """Morgan bit vectors ``(n, n_bits)`` uint8; unparsable SMILES give all-zero rows."""
    smiles = list(smiles)
    jobs = [(smiles[i : i + chunk], radius, n_bits) for i in range(0, len(smiles), chunk)]
    if not jobs:
        return np.zeros((0, n_bits), dtype=np.uint8)
    workers = workers or None
    if len(jobs) > 1 and workers != 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return np.concatenate(list(ex.map(_morgan_chunk, jobs)))
    return np.concatenate([_morgan_chunk(j) for j in jobs])
