"""Spectrum preprocessing and the binned sparse search library.

Preprocessing mirrors how the hidden test set was cleaned (peaks above precursor + 2 Da removed),
then keeps the ``TOP_N`` most intense peaks, square-root scales intensities and L2-normalises.
Spectra become sparse vectors over 0.01 Da bins; cosine similarity is a sparse dot product.
"""

from __future__ import annotations

import numpy as np
import pyarrow.compute as pc
import scipy.sparse as sp

BIN = 0.01
MAX_MZ = 2000.0
N_BINS = int(MAX_MZ / BIN) + 1
TOP_N = 64


def _flat_to_csr(offsets, mz, inten, precursor, top_n=TOP_N, min_rel=0.0):
    """Vectorised preprocessing of a flattened batch of spectra into CSR (rows = spectra)."""
    n = len(offsets) - 1
    lengths = np.diff(offsets)
    row = np.repeat(np.arange(n), lengths)
    keep = (mz <= np.repeat(precursor, lengths) + 2.0) & (mz < MAX_MZ) & (inten > min_rel) & np.isfinite(mz)
    row, mz, inten = row[keep], mz[keep], inten[keep]
    # top_n peaks per spectrum: sort by (row, -intensity), rank within row
    order = np.lexsort((-inten, row))
    row, mz, inten = row[order], mz[order], inten[order]
    starts = np.searchsorted(row, np.arange(n))
    rank = np.arange(len(row)) - starts[row]
    sel = rank < top_n
    row, mz, inten = row[sel], mz[sel], inten[sel]
    val = np.sqrt(inten).astype(np.float32)
    col = np.rint(mz / BIN).astype(np.int32)
    m = sp.csr_matrix((val, (row, col)), shape=(n, N_BINS), dtype=np.float32)  # duplicates in a bin are summed
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    return sp.diags((1.0 / norm).astype(np.float32)) @ m


def table_to_csr(table, precursor, top_n=TOP_N):
    """pyarrow table with ms2_mzs / ms2_normalized_intensities (aligned list columns) -> CSR."""
    mz_col = table.column("ms2_mzs").combine_chunks()
    in_col = table.column("ms2_normalized_intensities").combine_chunks()
    offsets = mz_col.offsets.to_numpy() - mz_col.offsets[0].as_py()
    mz = pc.list_flatten(mz_col).to_numpy(zero_copy_only=False).astype(np.float64)
    inten = pc.list_flatten(in_col).to_numpy(zero_copy_only=False).astype(np.float64)
    return _flat_to_csr(offsets, mz, inten, np.asarray(precursor, dtype=np.float64), top_n)


def arrays_to_csr(mzs, ints, precursor, top_n=TOP_N):
    lengths = np.array([len(a) for a in mzs])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    mz = np.concatenate(mzs).astype(np.float64) if len(mzs) else np.zeros(0)
    inten = np.concatenate(ints).astype(np.float64) if len(ints) else np.zeros(0)
    return _flat_to_csr(offsets, mz, inten, np.asarray(precursor, dtype=np.float64), top_n)


def widen(q: sp.csr_matrix) -> sp.csr_matrix:
    """Spread query peaks to the neighbouring bins (about +-0.015 Da tolerance) for matching."""
    return (q + _shift(q, 1) + _shift(q, -1)).tocsr()


def _shift(q, k):
    q = q.tocoo()
    col = q.col + k
    ok = (col >= 0) & (col < q.shape[1])
    return sp.csr_matrix((q.data[ok], (q.row[ok], col[ok])), shape=q.shape)
