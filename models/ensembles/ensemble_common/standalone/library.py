"""Cosine library search over per-structure spectrum sums (direction == centroid direction)."""

from __future__ import annotations

import numpy as np


class CentroidLibrary:
    def __init__(self, n_structures: int, dim: int):
        self.sums = np.zeros((n_structures, dim), dtype=np.float32)
        self.counts = np.zeros(n_structures, dtype=np.int32)

    def add(self, struct_idx: int, vec: np.ndarray) -> None:
        self.sums[struct_idx] += vec
        self.counts[struct_idx] += 1

    def scores(self, qvec: np.ndarray, cand_idx: np.ndarray, *, absent: np.ndarray | None = None,
               subtract: tuple[int, np.ndarray, int] | None = None) -> np.ndarray:
        """Cosine of ``qvec`` to each candidate's centroid.

        ``absent`` (bool per structure): candidates with no library evidence score 0 (held-out structures).
        ``subtract`` = (struct_idx, summed_vectors, n): remove the query's own spectra from that
        structure's sum (leave-query-out) so a covered query cannot see itself.
        """
        c = self.sums[cand_idx].copy()
        counts = self.counts[cand_idx].copy()
        if subtract is not None:
            s_idx, vec_sum, n = subtract
            hit = np.nonzero(cand_idx == s_idx)[0]
            if hit.size:
                c[hit[0]] -= vec_sum
                counts[hit[0]] -= n
        norms = np.linalg.norm(c, axis=1)
        sims = np.divide(c @ qvec, norms, out=np.zeros(len(c), dtype=np.float32), where=norms > 1e-8)
        sims[counts <= 0] = 0.0
        if absent is not None:
            sims[absent[cand_idx]] = 0.0
        return sims
