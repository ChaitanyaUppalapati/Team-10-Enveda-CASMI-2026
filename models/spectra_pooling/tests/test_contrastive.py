"""Member 7 checks for Architecture 3 (contrastive peak transformer) and its reranker features."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="contrastive tests require the optional ml dependencies")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import contrastive as ct  # noqa: E402
from common.rerank import CLIP_FEATURES, clip_features  # noqa: E402


def test_top_peaks_keeps_most_intense_sorted_by_mz():
    mz = np.array([[50.0, 60.0, 70.0, 80.0, 0.0]], dtype=np.float32)
    it = np.array([[0.1, 0.9, 0.5, 0.7, 0.0]], dtype=np.float32)
    m2, i2 = ct.top_peaks(mz, it, n=3)
    assert m2.tolist() == [[60.0, 70.0, 80.0]]
    assert np.allclose(i2, [[0.9, 0.5, 0.7]])
    m3, i3 = ct.top_peaks(np.array([[50.0, 0, 0, 0]]), np.array([[1.0, 0, 0, 0]]), n=3)
    assert m3[0, 0] == 50.0 and (i3[0, 1:] == 0).all() and (m3[0, 1:] == 0).all()


def test_window_bounds_and_negatives_stay_in_window_and_skip_positive():
    pool_mass = np.array([100.0, 200.0, 200.0005, 200.001, 300.0])
    pos = np.array([2, 0])
    lo, hi = ct.window_bounds(pool_mass, pool_mass[pos], ppm=10.0)
    assert (lo[0], hi[0]) == (1, 4) and (lo[1], hi[1]) == (0, 1)
    neg = ct.sample_negatives(np.random.default_rng(0), pos, lo, hi, n_neg=50, n_pool=len(pool_mass))
    assert set(neg[0]) <= {1, 3}                 # same window, never the positive
    assert 0 not in set(neg[1])                  # lonely window: random others, not the positive


def test_gated_attention_starts_as_mean_and_respects_mask():
    pool = ct.GatedAttentionPool(8)
    h = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    att, _ = pool(h, mask)
    mean, _ = pool(h, mask, mode="mean")
    assert torch.allclose(att, mean, atol=1e-6)
    h2 = h.clone()
    h2[0, 2] = 100.0  # padded row must not matter
    assert torch.allclose(pool(h2, mask)[0][0], att[0], atol=1e-6)


def test_retriever_shapes_and_loss_backprop():
    torch.manual_seed(0)
    model = ct.Retriever(d=32, layers=1, heads=4, emb=16, fp_bits=64)
    n, p = 5, 6
    mz = torch.rand(n, p) * 300 + 50
    it = torch.rand(n, p)
    it[0, 3:] = 0  # padded peaks
    h = model.spectra(mz, it, torch.full((n,), 400.0), torch.zeros(n, dtype=torch.long))
    assert h.shape == (n, 32)
    hs, mask = ct.pad_sets([np.array([0, 1]), np.array([2, 3, 4])], h)
    q, fp_logits, att = model.query(hs, mask)
    assert q.shape == (2, 16) and fp_logits.shape == (2, 64) and att.shape == (2, 3)
    assert torch.allclose(q.norm(dim=-1), torch.ones(2), atol=1e-5)
    m = model.molecules(torch.randint(0, 2, (2 * 4, 64)).float()).view(2, 4, -1)
    loss = ct.info_nce(q, m[:, 0], m[:, 1:], model.scale())
    loss.backward()
    assert torch.isfinite(loss) and model.spectra.peak[0].weight.grad is not None


def test_peak_order_does_not_matter():
    torch.manual_seed(1)
    enc = ct.PeakEncoder(d=16, layers=1, heads=2, dropout=0.0).eval()
    mz = torch.tensor([[60.0, 90.0, 120.0]])
    it = torch.tensor([[0.5, 0.2, 0.8]])
    a = enc(mz, it, torch.tensor([300.0]), torch.tensor([1]))
    b = enc(mz[:, [2, 0, 1]], it[:, [2, 0, 1]], torch.tensor([300.0]), torch.tensor([1]))
    assert torch.allclose(a, b, atol=1e-5)


def test_training_improves_on_a_learnable_toy_task():
    """Two structures per mass window, each with its own peak: the model must learn to tell them apart."""
    rng = np.random.default_rng(0)
    n_struct, bits = 40, 64
    pool_mass = np.repeat(np.arange(n_struct // 2) * 10.0 + 200.0, 2)
    pool_mass[1::2] += 0.0005  # isomer pairs, inside 10 ppm
    fp = np.packbits(rng.random((n_struct, bits)) < 0.3, axis=1)
    keys = [f"k{i}" for i in range(n_struct)]
    key_index = {k: i for i, k in enumerate(keys)}
    rows, mz, it = [], [], []
    for i in range(n_struct):
        for _ in range(3):
            rows.append(i)
            mz.append([50.0 + i * 3.0, 0, 0, 0])
            it.append([1.0, 0, 0, 0])
    rows = np.array(rows)
    spec = {"mz": np.array(mz, np.float32), "intensity": np.array(it, np.float32),
            "precursor": pool_mass[rows].astype(np.float32) + 1.0, "adduct": np.zeros(len(rows), np.int64)}
    groups = {k: np.flatnonzero(rows == i) for i, k in enumerate(keys)}

    def score(model):
        H = ct.spectrum_vectors(model, spec["mz"], spec["intensity"], spec["precursor"], spec["adduct"], "cpu")
        q, _ = ct.query_vectors(model, [groups[k] for k in keys], H, "cpu")
        sims = ct.window_scores(model, fp, pool_mass, pd.Series(pool_mass, index=keys), q, "cpu")
        return float(np.mean([sims[k][0][np.argmax(sims[k][1])] == key_index[k] for k in keys]))

    model = ct.Retriever(d=32, layers=1, heads=4, emb=16, fp_bits=bits, dropout=0.0)
    before = score(model.eval())
    model, hist = ct.train_retriever(model, spec, groups, keys, key_index, fp, pool_mass, "cpu", epochs=30,
                                     batch=8, n_neg=3, lr=3e-3, max_set=3, score_fn=score, log=lambda r: None)
    assert hist.val_mrr.max() >= 0.9 > before


def test_clip_features_rank_and_isomer_gap():
    t = clip_features([0.2, 0.9, 0.5], ["C6H6", "C6H6", "C7H8"])
    assert list(t.columns) == CLIP_FEATURES
    assert t.clip_rank.tolist() == [3, 1, 2]
    assert np.allclose(t.clip_iso_gap, [-0.7, 0.0, 0.0])
    assert np.allclose(t.clip_gap, [-0.7, 0.0, -0.4])


def test_keep_prob_zero_molecules_never_train():
    """keep_prob 0 keeps a molecule out of every epoch; the run still finishes and anneals."""
    rng = np.random.default_rng(1)
    keys = [f"k{i}" for i in range(16)]
    pool_mass = np.arange(16) * 10.0 + 200.0
    fp = np.packbits(rng.random((16, 64)) < 0.3, axis=1)
    spec = {"mz": np.full((16, 4), 80.0, np.float32), "intensity": np.ones((16, 4), np.float32),
            "precursor": (pool_mass + 1).astype(np.float32), "adduct": np.zeros(16, np.int64)}
    seen = []
    model = ct.Retriever(d=16, layers=1, heads=2, emb=8, fp_bits=64, dropout=0.0)
    orig = ct.sample_sets

    def spy(groups, ks, rng_, max_set=8):
        seen.extend(ks)
        return orig(groups, ks, rng_, max_set=max_set)

    ct.sample_sets = spy
    try:
        keep = np.r_[np.ones(8), np.zeros(8)]
        model, hist = ct.train_retriever(model, spec, {k: np.array([i]) for i, k in enumerate(keys)}, keys,
                                         {k: i for i, k in enumerate(keys)}, fp, pool_mass, "cpu", epochs=3,
                                         batch=4, n_neg=2, keep_prob=keep, log=lambda r: None)
    finally:
        ct.sample_sets = orig
    assert set(seen) <= set(keys[:8]) and len(seen) > 0
    assert len(hist) == 3
