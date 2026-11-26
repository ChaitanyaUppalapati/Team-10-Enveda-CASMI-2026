"""Member 7 checks: pooling shapes, masking, permutation invariance, sampling and scoring."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="pooling tests require the optional ml dependencies")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluate import mrr_at_k, ranks_of_truth, score_queries, tanimoto_scores  # noqa: E402
from common.pooling import AttentionPool, DeepSetsPool, SpectrumEncoder, build_pool, mean_probabilities  # noqa: E402
from common.sets import draw_set_sizes, group_indices, pad_sets, sample_sets  # noqa: E402

EMB, FP = 16, 32


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def batch(sizes):
    sets = []
    start = 0
    for n in sizes:
        sets.append(np.arange(start, start + n))
        start += n
    feats = torch.randn(start, EMB)
    return pad_sets(sets, feats), feats, sets


@pytest.mark.parametrize("name", ["deepsets", "attention"])
def test_pool_shapes_and_any_set_size(name):
    pool = build_pool(name, EMB, FP, hidden=24).eval()
    (h, mask), _, _ = batch([1, 3, 16])
    logits, weights = pool(h, mask)
    assert logits.shape == (3, FP)
    if name == "attention":
        assert torch.allclose(weights.sum(1), torch.ones(3), atol=1e-6)
        assert (weights[~mask] == 0).all()


@pytest.mark.parametrize("cls", [DeepSetsPool, AttentionPool])
def test_padding_and_order_do_not_change_output(cls):
    pool = cls(EMB, FP, hidden=24).eval()
    feats = torch.randn(4, EMB)
    alone, _ = pool(*pad_sets([np.arange(4)], feats))
    padded, _ = pool(*pad_sets([np.arange(4), np.arange(1)], feats))
    shuffled, _ = pool(*pad_sets([np.array([3, 1, 0, 2])], feats))
    assert torch.allclose(alone[0], padded[0], atol=1e-5)
    assert torch.allclose(alone[0], shuffled[0], atol=1e-5)


def test_mean_probabilities_ignores_padding():
    probs = torch.tensor([[[0.2, 0.4], [0.6, 0.8], [9.0, 9.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(mean_probabilities(probs, mask), torch.tensor([[0.4, 0.6]]))


def test_encoder_shapes():
    enc = SpectrumEncoder(n_bins=50, n_meta=3, fp_size=FP, hidden=20, emb=EMB)
    x, meta = torch.zeros(5, 50), torch.zeros(5, 3)
    assert enc.embed(x, meta).shape == (5, EMB) and enc(x, meta).shape == (5, FP)


def test_set_sampling_is_test_like_and_within_groups():
    rng = np.random.default_rng(0)
    sizes = draw_set_sizes(rng, 20000)
    assert sizes.min() == 1 and sizes.max() <= 16 and np.median(sizes) == 3
    groups = group_indices(np.array(["b", "a", "b", "a", "b"]))
    assert set(groups) == {"a", "b"} and sorted(groups["b"]) == [0, 2, 4]
    for key, rows in zip(["a", "b"], sample_sets(groups, ["a", "b"], rng), strict=True):
        assert set(rows) <= set(groups[key]) and len(set(rows)) == len(rows)


def test_scoring_and_mrr():
    fps = {"AAAAAAAAAAAAAA": np.array([1, 1, 0, 0]), "BBBBBBBBBBBBBB": np.array([0, 0, 1, 1])}
    pred = np.array([[0.9, 0.8, 0.1, 0.0], [0.9, 0.8, 0.1, 0.0]])
    assert tanimoto_scores(pred[0], np.stack(list(fps.values())))[0] > 0.7
    scores = score_queries(pred, ["q1", "q2"], {"q1": list(fps), "q2": list(fps)}, fps)
    ranks = ranks_of_truth(scores, {"q1": "AAAAAAAAAAAAAA", "q2": "BBBBBBBBBBBBBB"})
    assert ranks.tolist() == [1.0, 2.0]
    assert mrr_at_k(ranks) == pytest.approx(0.75)
    assert mrr_at_k(pd.Series([np.nan, 30.0])) == 0.0


def test_tuned_mean_control_trains_a_copy_of_a_frozen_head():
    from common.full import TunedMean

    head = torch.nn.Linear(EMB, FP)
    for p in head.parameters():
        p.requires_grad_(False)
    control = TunedMean(head)
    (h, mask), _, _ = batch([1, 3])
    logits, _ = control(h, mask)
    logits.sum().backward()
    assert control.head.weight.grad is not None and head.weight.grad is None
    probs = torch.sigmoid(head(h))
    expected = (probs[1, :3]).mean(0)
    assert torch.allclose(torch.sigmoid(logits[1]), expected, atol=1e-4)


def test_paired_bootstrap_detects_a_clear_difference():
    from common.full import paired_bootstrap

    rng = np.random.default_rng(0)
    b = rng.random(500)
    out = paired_bootstrap(b + 0.1, b)
    assert out["delta"] == pytest.approx(0.1) and out["ci_low"] > 0
    same = paired_bootstrap(b, b)
    assert same["delta"] == 0 and same["ci_low"] <= 0 <= same["ci_high"]


def test_attention_head_starts_as_tuned_mean_and_fusion_orders_by_rank():
    from common.full import AttentionHead, TunedMean, rrf_fuse

    head = torch.nn.Linear(EMB, FP)
    (h, mask), _, _ = batch([2, 4])
    att, _ = AttentionHead(head, EMB)(h, mask)
    avg, _ = TunedMean(head)(h, mask)
    assert torch.allclose(att, avg, atol=1e-4)
    assert rrf_fuse([[1, 2, 3], [1, 3, 2]]) [:1] == [1]


# ---- v2: external candidates and gated B2 ----
from common.candidates import gate, smiles_column  # noqa: E402


def test_smiles_column_prefers_canonical_and_falls_back():
    assert smiles_column(["id", "SMILES", "canonical_smiles"]) == "canonical_smiles"
    assert smiles_column(["Identifier", "SMILES", "InChIKey"]) == "SMILES"
    assert smiles_column(["x", "my_smiles_str"]) == "my_smiles_str"
    with pytest.raises(ValueError):
        smiles_column(["a", "b"])


def test_gate_switches_only_above_tau():
    model = {"a": [1, 2], "b": [3, 4]}
    alt = {"a": [9, 8], "b": [7, 6]}
    out = gate(model, alt, {"a": 0.9, "b": 0.1}, tau=0.5)
    assert out == {"a": [9, 8], "b": [3, 4]}
    assert gate(model, alt, {"a": 0.9}, tau=float("inf")) == model
