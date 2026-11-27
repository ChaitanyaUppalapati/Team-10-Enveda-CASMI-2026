"""Tests for the contrastive model's loss, hard-negative sampler and ranking. Owner: Member 4."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from sf_common import contrastive as C  # noqa: E402
from sf_common import features as F  # noqa: E402


def _unit(*rows):
    t = torch.tensor(rows, dtype=torch.float32)
    return t / t.norm(dim=-1, keepdim=True)


def test_loss_is_low_when_spectra_match_their_molecule_and_high_otherwise():
    spec = _unit([1, 0], [0, 1])
    mol_hard = _unit([[0.7, 0.7]], [[0.7, 0.7]]).reshape(2, 1, 2)
    ids, hard_ids = torch.tensor([0, 1]), torch.tensor([[2], [2]])
    scale = torch.tensor(20.0)
    good = C.contrastive_loss(spec, spec.clone(), mol_hard, ids, hard_ids, scale)
    bad = C.contrastive_loss(spec, spec.flip(0), mol_hard, ids, hard_ids, scale)
    assert good < 0.1 < bad


def test_loss_masks_duplicate_structures_and_hard_negatives_equal_to_the_positive():
    spec = _unit([1, 0], [1, 0])
    scale = torch.tensor(10.0)
    # both spectra belong to structure 5: the other item must not count as a negative
    same = C.contrastive_loss(spec, spec.clone(), _unit([[0, 1]], [[0, 1]]).reshape(2, 1, 2),
                              torch.tensor([5, 5]), torch.tensor([[7], [7]]), scale)  # fmt: skip
    assert same.item() < 1e-3
    # a hard negative that IS the positive structure must be masked too
    clash = C.contrastive_loss(spec[:1], spec[:1].clone(), spec[:1].reshape(1, 1, 2),
                               torch.tensor([5]), torch.tensor([[5]]), scale)  # fmt: skip
    assert clash.item() < 1e-3


def test_sample_hard_negatives_stay_in_the_mass_window():
    cluster = [300.0, 300.0001, 300.0002, 300.0003]
    pool_mass = np.concatenate([np.linspace(100, 200, 50), cluster, np.linspace(400, 500, 50)])
    pos = np.array([51, 3])  # a structure in the dense 300 cluster, and one with no same-mass neighbours
    rng = np.random.default_rng(0)
    hard = C.sample_hard_negatives(pool_mass, pos, k=16, ppm=10.0, rng=rng)
    assert hard.shape == (2, 16)
    assert np.all(np.abs(pool_mass[hard[0]] - 300.0) < 300.0 * 10e-6 + 1e-9)
    assert np.all((hard[1] >= 0) & (hard[1] < len(pool_mass)))  # fallback: neighbours in mass order


def test_aggregate_embeddings_averages_per_molecule_and_renormalises():
    emb = np.array([[1, 0], [0, 1], [0, 1]], dtype=np.float32)
    ids, mean = C.aggregate_embeddings(["a", "a", "b"], emb)
    assert ids == ["a", "b"]
    np.testing.assert_allclose(mean[0], [0.7071, 0.7071], atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(mean, axis=1), 1.0, atol=1e-5)


def test_network_shapes_and_unit_norm():
    net = C.ContrastiveNet(n_bits=64, d=16, spec_dim=32, hidden=48, mol_hidden=40)
    s = net.encode_spectrum(torch.zeros(3, F.N_BINS), torch.zeros(3, F.N_META))
    m = net.encode_molecule(torch.ones(5, 64))
    assert s.shape == (3, 16) and m.shape == (5, 16)
    torch.testing.assert_close(s.norm(dim=-1), torch.ones(3), atol=1e-4, rtol=0)


def test_rank_contrastive_orders_by_similarity_within_the_mass_window():
    class FakeCP:
        @staticmethod
        def window_indices(pool_mass, mass, ppm):
            tol = mass * ppm * 1e-6
            lo = np.searchsorted(pool_mass, mass - tol)
            return np.arange(lo, np.searchsorted(pool_mass, mass + tol, side="right"))

        @staticmethod
        def fill_to_k(ranked, pool_mass, mass, k=25):
            return [int(i) for i in ranked][:k]

    pool = pd.DataFrame({"mass": [100.0, 200.0, 200.0, 200.0, 300.0]})
    pool_emb = np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8], [1, 0]], dtype=np.float16)
    query = np.array([[0.0, 1.0]], dtype=np.float32)
    out = C.rank_contrastive(FakeCP, pool, pool_emb, ["m"], query, {"m": 200.0}, 10.0)
    assert out["m"] == [1, 3, 2]  # only the three 200 Da structures, most similar first


def test_rank_contrastive_prior_can_break_a_near_tie_but_not_override_a_clear_gap():
    class FakeCP:
        @staticmethod
        def window_indices(pool_mass, mass, ppm):
            return np.arange(len(pool_mass))

        @staticmethod
        def fill_to_k(ranked, pool_mass, mass, k=25):
            return [int(i) for i in ranked][:k]

    pool = pd.DataFrame({"mass": [200.0, 200.0, 200.0]})
    pool_emb = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0]], dtype=np.float16)  # cosines 1.0, 0.99, 0.0
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    prior = np.array([0.0, 3.0, 9.0], dtype=np.float32)
    plain = C.rank_contrastive(FakeCP, pool, pool_emb, ["m"], q, {"m": 200.0}, 10.0)
    args = (FakeCP, pool, pool_emb, ["m"], q, {"m": 200.0}, 10.0)
    boosted = C.rank_contrastive(*args, prior, prior_weight=1.0, scale=70.0)
    assert plain["m"] == [0, 1, 2]
    assert boosted["m"][:2] == [1, 0]  # 70*0.99 + 3 > 70*1.0 + 0: the prior reorders the near tie
    assert boosted["m"][-1] == 2  # but a candidate that is far off in cosine stays last despite a large prior


def test_prior_topk_only_reorders_inside_the_head_of_the_list():
    class FakeCP:
        @staticmethod
        def window_indices(pool_mass, mass, ppm):
            return np.arange(len(pool_mass))

        @staticmethod
        def fill_to_k(ranked, pool_mass, mass, k=25):
            return [int(i) for i in ranked][:k]

    pool = pd.DataFrame({"mass": [200.0] * 4})
    pool_emb = np.array([[1.0, 0.0], [0.99, 0.141], [0.9, 0.436], [0.8, 0.6]], dtype=np.float16)
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    prior = np.array([0.0, 0.0, 0.0, 50.0], dtype=np.float32)  # a huge prior on the worst candidate
    args = (FakeCP, pool, pool_emb, ["m"], q, {"m": 200.0}, 10.0)
    assert C.rank_contrastive(*args, prior, 1.0, 70.0)["m"][0] == 3  # unrestricted: the prior promotes it to rank 1
    assert C.rank_contrastive(*args, prior, 1.0, 70.0, prior_topk=2)["m"] == [0, 1, 2, 3]  # outside the head: untouched
