"""Gated library-search (B2) fusion for the model's rankings. Owner: Member 4.

Library search is near-perfect when a structure has reference spectra and useless otherwise, so B2's
ranking replaces the model's only for molecules whose best library match scores at least ``tau``
(Member 7's gate; ``b2_top_scores``, ``gate`` and ``tune_gate`` are loaded from
``models/spectra_pooling/common/candidates.py``). ``tau`` and the alternative ranking are chosen on
validation, and the honest estimate is the 2-fold cross-fitted score.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
M7_CANDIDATES = ROOT / "models" / "spectra_pooling" / "common" / "candidates.py"


def load_m7():
    """Import Member 7's candidates/gate module by path (it only needs numpy and pandas)."""
    if "m7_candidates" in sys.modules:
        return sys.modules["m7_candidates"]
    if not M7_CANDIDATES.is_file():
        raise FileNotFoundError(f"{M7_CANDIDATES} not found; it must be part of the code dataset")
    spec = importlib.util.spec_from_file_location("m7_candidates", M7_CANDIDATES)
    module = importlib.util.module_from_spec(spec)
    sys.modules["m7_candidates"] = module
    spec.loader.exec_module(module)
    return module


def rrf_fuse(rank_lists, weights=None, k=60, top=25):
    """Reciprocal-rank fusion of ranked lists of pool indices (best first); same as Member 7's."""
    weights = weights or [1.0] * len(rank_lists)
    score: dict[int, float] = {}
    for lst, w in zip(rank_lists, weights, strict=True):
        for r, idx in enumerate(lst):
            score[int(idx)] = score.get(int(idx), 0.0) + w / (k + r + 1)
    return sorted(score, key=lambda i: (-score[i], i))[:top]


def fuse_with_b2(proto, val_rank: dict, test_rank: dict, ppm: float = 10.0, log=print):
    """Run B2 on validation and test, tune the gate on validation, return gated rankings and info.

    ``val_rank`` / ``test_rank`` map molecule_id to pool indices (best first), as produced by
    ``casmi_pipeline.rank_by_fingerprint``.
    """
    cp, m7 = proto.cp, load_m7()
    pool, pool_keys = proto.pool, proto.pool.key.to_numpy()
    va_mz, va_in = cp.read_spectra(proto.train_path, proto.queries.row_id.to_numpy())
    te_mz, te_in = cp.read_spectra(proto.test_path, proto.test.row_id.to_numpy())
    val_mass, test_mass = cp.consensus_masses(proto.queries), cp.consensus_masses(proto.test)

    lib = cp.SpectralLibrary(proto.data_dir, proto.meta, proto.lib_mask, pool, val_mass.to_numpy(), ppm=ppm)
    val_b2 = cp.run_b2(pool, lib, proto.queries, va_mz, va_in, ppm=ppm, pooling="mean")
    val_score = m7.b2_top_scores(cp, pool, lib, proto.queries, va_mz, va_in, val_b2)
    log(f"B2 validation done ({len(val_b2)} molecules)")
    lib = cp.SpectralLibrary(proto.data_dir, proto.meta, proto.lib_mask, pool, test_mass.to_numpy(), ppm=ppm)
    test_b2 = cp.run_b2(pool, lib, proto.test, te_mz, te_in, ppm=ppm, pooling="mean")
    test_score = m7.b2_top_scores(cp, pool, lib, proto.test, te_mz, te_in, test_b2)
    log(f"B2 test done ({len(test_b2)} molecules)")
    del lib

    fused_val = {m: rrf_fuse([val_rank[m], val_b2[m]], [1.0, 2.0]) for m in val_rank}
    fused_test = {m: rrf_fuse([test_rank[m], test_b2[m]], [1.0, 2.0]) for m in test_rank}
    alts_val = {"B2": val_b2, "fusion(B2 x2.0)": fused_val}
    alts_test = {"B2": test_b2, "fusion(B2 x2.0)": fused_test}
    info = m7.tune_gate(cp, proto.truth, pool_keys, val_rank, alts_val, val_score)
    val_gated = m7.gate(val_rank, alts_val[info["alt"]], val_score, info["tau"])
    test_gated = m7.gate(test_rank, alts_test[info["alt"]], test_score, info["tau"])
    info["share_gated_test"] = round(float(np.mean([test_score[m] >= info["tau"] for m in test_rank])), 4)
    return val_gated, test_gated, info, {"B2": val_b2, "scores": val_score}
