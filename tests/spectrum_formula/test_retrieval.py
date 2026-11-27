"""Tests for analog-spectrum retrieval. Owner: Member 4."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from sf_common import retrieval as R  # noqa: E402


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_knn_returns_best_first_and_matches_brute_force_across_chunks():
    rng = np.random.default_rng(0)
    lib = _unit(rng.normal(size=(50, 8)))
    q = _unit(rng.normal(size=(5, 8)))
    sims, idx = R.knn(q, lib.astype(np.float16), k=4, device="cpu", q_batch=2, l_chunk=13)  # forces merging
    brute = (q @ lib.T)
    for row in range(5):
        expected = np.argsort(-brute[row])[:4]
        assert idx[row].tolist() == expected.tolist()
        np.testing.assert_allclose(sims[row], brute[row][expected], atol=2e-3)


def test_consensus_merges_spectra_per_structure_and_weights_by_similarity():
    # two library spectra of structure 0 (very close), one of structure 1 (far)
    pool_fp = np.packbits(np.array([[1, 1, 0, 0] + [0] * 4, [0, 0, 1, 1] + [0] * 4], dtype=np.uint8), axis=1)
    lib_struct = np.array([0, 0, 1])
    sims = np.array([[0.95, 0.94, 0.10]])
    idx = np.array([[0, 1, 2]])
    consensus, tops = R.consensus_fingerprints(np.array(["m"]), sims, idx, lib_struct, pool_fp, k=3, beta=20.0)
    c = consensus["m"]
    assert c.shape == (8,) and c[:2].min() > 0.99 and c[2:4].max() < 0.01  # structure 0 dominates
    assert tops["m"][0][0] == 0 and tops["m"][0][1] > 0.99


def test_consensus_pools_all_spectra_of_a_molecule():
    pool_fp = np.packbits(np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8), axis=1)
    lib_struct = np.array([0, 1])
    sims = np.array([[0.9], [0.9]])  # molecule "m" has two spectra, one neighbour each
    idx = np.array([[0], [1]])
    consensus, _ = R.consensus_fingerprints(np.array(["m", "m"]), sims, idx, lib_struct, pool_fp, k=1)
    np.testing.assert_allclose(consensus["m"][:2], [0.5, 0.5], atol=1e-6)


def test_soft_tanimoto_prefers_the_matching_candidate():
    cons = np.array([0.9, 0.9, 0.0, 0.0], dtype=np.float32)
    bits = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    sims = R.soft_tanimoto(cons, bits)
    assert sims[0] > 0.7 > sims[1]


def test_rank_with_retrieval_lets_the_consensus_reorder_near_ties_only():
    class FakeCP:
        @staticmethod
        def window_indices(pool_mass, mass, ppm):
            return np.arange(len(pool_mass))

        @staticmethod
        def fill_to_k(ranked, pool_mass, mass, k=25):
            return [int(i) for i in ranked][:k]

    pool = pd.DataFrame({"mass": [200.0, 200.0, 200.0]})
    pool_emb = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0]], dtype=np.float16)  # cosines 1.0, 0.99, 0.0
    bits = [[0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]
    fp = np.packbits(np.array(bits, dtype=np.uint8), axis=1)
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    consensus = {"m": np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)}  # looks like candidates 1 and 2
    args = (FakeCP, pool, pool_emb, fp, ["m"], q, {"m": 200.0})
    assert R.rank_with_retrieval(*args, consensus, 10.0, lam=0.0, scale=70.0)["m"] == [0, 1, 2]
    out = R.rank_with_retrieval(*args, consensus, 10.0, lam=20.0, scale=70.0)["m"]
    assert out[:2] == [1, 0]  # candidate 1 is nearly as close in cosine and matches the neighbours' structure
    assert out[-1] == 2  # candidate 2 matches the neighbours too but is far in cosine: stays last
