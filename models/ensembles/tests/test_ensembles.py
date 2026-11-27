"""Member 8 checks on synthetic score files: normalisation, fusion, stacking, end-to-end CLI."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ensemble_common.fusion import fuse, tune_weights  # noqa: E402
from ensemble_common.inputs import align_scores  # noqa: E402
from ensemble_common.metrics import mrr_at_k, reciprocal_ranks  # noqa: E402
from ensemble_common.normalize import normalize  # noqa: E402
from ensemble_common.stacking import add_labels, build_features, out_of_fold_scores  # noqa: E402


def key(i: int) -> str:
    """Valid synthetic InChIKey14: 14 uppercase letters."""
    return (chr(65 + i // 26) + chr(65 + i % 26)).ljust(14, "A")


def make_world(n_q=40, n_c=12, seed=0):
    """Two models: A good on even queries, B good on odd ones; truth is candidate 0."""
    rng = np.random.default_rng(seed)
    rows_a, rows_b, truth = [], [], []
    for q in range(n_q):
        truth.append((f"q{q}", key(0)))
        for c in range(n_c):
            good = c == 0
            a = rng.normal() + (3.0 if good and q % 2 == 0 else 0.0)
            b = rng.normal() * 10 + (30.0 if good and q % 2 == 1 else 0.0)
            rows_a.append((f"q{q}", key(c), a))
            rows_b.append((f"q{q}", key(c), b))
    cols = ["query_id", "inchikey14", "score"]
    return pd.DataFrame(rows_a, columns=cols), pd.DataFrame(rows_b, columns=cols), pd.DataFrame(truth, columns=cols[:2])


def test_reciprocal_rank_and_cutoff():
    s = pd.DataFrame({"query_id": ["q"] * 3, "inchikey14": [key(0), key(1), key(2)], "score": [0.1, 0.9, 0.5]})
    truth = pd.DataFrame({"query_id": ["q"], "inchikey14": [key(2)]})
    assert reciprocal_ranks(s, truth)["q"] == pytest.approx(0.5)
    assert reciprocal_ranks(s, truth, k=1)["q"] == 0.0
    assert mrr_at_k(s, pd.DataFrame({"query_id": ["q"], "inchikey14": [key(9)]})) == 0.0


def test_align_keeps_unscored_candidates_as_nan():
    a, b, _ = make_world(2, 3)
    wide = align_scores({"A": a, "B": b.iloc[1:]})
    assert wide["B"].isna().sum() == 1 and len(wide) == len(a)


@pytest.mark.parametrize("method", ["rrf", "minmax", "zscore"])
def test_normalize_is_per_query_and_scale_free(method):
    a, _, _ = make_world(5, 8)
    scaled = a.assign(score=a["score"] * 1000 + 7)
    n1 = normalize(align_scores({"A": a}), ["A"], method)
    n2 = normalize(align_scores({"A": scaled}), ["A"], method)
    np.testing.assert_allclose(n1["A"], n2["A"], atol=1e-9)
    assert np.isfinite(n1["A"]).all()


def test_missing_candidate_never_beats_scored_one():
    a, b, _ = make_world(3, 6)
    wide = align_scores({"A": a, "B": b[b["inchikey14"] != key(0)]})
    for method in ("rrf", "minmax", "zscore"):
        n = normalize(wide, ["A", "B"], method)
        miss = wide["B"].isna()
        assert n.loc[miss, "B"].max() <= n.loc[~miss, "B"].min() + 1e-12


def test_constant_scores_do_not_nan():
    df = pd.DataFrame({"query_id": ["q"] * 3, "inchikey14": [key(i) for i in range(3)], "score": [1.0, 1.0, 1.0]})
    for method in ("rrf", "minmax", "zscore"):
        assert np.isfinite(normalize(align_scores({"A": df}), ["A"], method)["A"]).all()


def test_fusion_beats_each_model_when_they_are_complementary():
    a, b, truth = make_world()
    ids = ["A", "B"]
    norm = normalize(align_scores({"A": a, "B": b}), ids, "zscore")
    weights, best = tune_weights(norm, truth, ids, n_random=60)
    solo = [mrr_at_k(fuse(norm, {m: 1.0}), truth) for m in ids]
    assert best > max(solo)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_out_of_fold_fusion_never_uses_its_own_labels():
    """Each fold's weights are tuned without that fold's truth: changing a held-out label cannot move them."""
    from ensemble_common.fusion import out_of_fold_fusion, query_folds

    a, b, truth = make_world(40, 10)
    ids = ["A", "B"]
    norm = normalize(align_scores({"A": a, "B": b}), ids, "zscore")
    oof, fold_w = out_of_fold_fusion(norm, truth, ids, n_splits=4, n_random=30)
    assert len(oof) == len(norm) and oof.index.equals(norm.index) and len(fold_w) == 4
    folds = query_folds(norm["query_id"], 4)
    q0 = folds[folds == 0].index
    wrong = truth.assign(inchikey14=np.where(truth.query_id.isin(q0), key(5), truth.inchikey14))
    _, fold_w2 = out_of_fold_fusion(norm, wrong, ids, n_splits=4, n_random=30)
    assert fold_w2[0] == fold_w[0]
    assert mrr_at_k(oof, truth) > max(mrr_at_k(a, truth), mrr_at_k(b, truth)) - 0.05


def test_stacker_oof_is_leak_free_and_competitive():
    a, b, truth = make_world(60, 12)
    ids = ["A", "B"]
    feats = build_features(align_scores({"A": a, "B": b}), ids)
    y = add_labels(feats, truth)
    assert y.sum() == 60
    pytest.importorskip("sklearn")
    oof = out_of_fold_scores(feats, y, "logreg", n_splits=4)
    assert len(oof) == len(feats) and np.isfinite(oof["score"]).all()
    solo = max(mrr_at_k(a, truth), mrr_at_k(b, truth))
    assert mrr_at_k(oof, truth) >= solo - 0.05


def test_stacker_lightgbm_runs():
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError) as exc:  # OSError: missing libomp on macOS (brew install libomp)
        pytest.skip(f"lightgbm unavailable: {exc.__class__.__name__}")
    a, b, truth = make_world(40, 10)
    feats = build_features(align_scores({"A": a, "B": b}), ["A", "B"])
    oof = out_of_fold_scores(feats, add_labels(feats, truth), "lgbm_rank", n_splits=3)
    assert mrr_at_k(oof, truth) > 0.2


def test_end_to_end_cli(tmp_path, monkeypatch):
    """train + predict for both architectures through the real entry points and score files."""
    import importlib.util

    from casmi.interfaces import read_scores, write_scores

    monkeypatch.setenv("CASMI_OUTPUT_DIR", str(tmp_path))
    a, b, truth = make_world(30, 10)
    for split in ("val", "test"):
        write_scores(a, "m.a", split)
        write_scores(b, "m.b", split)
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path)

    def load(arch, name):
        p = Path(__file__).resolve().parents[1] / arch / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{arch}_{name}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    import yaml

    variants = (
        ("arch1_late_fusion", {"n_random": 30}),
        ("arch2_recall_priors", {"meta": "logreg", "n_splits": 3}),
    )
    for arch, extra in variants:
        cfg = tmp_path / f"{arch}.yaml"
        cfg.write_text(yaml.safe_dump({"model_id": f"ensembles.{arch}", "inputs": ["m.a", "m.b"],
                                       "truth": str(truth_path), **extra}))
        load(arch, "train").main(["--config", str(cfg)])
        for split in ("val", "test"):
            load(arch, "predict").main(["--config", str(cfg), "--split", split])
            out = read_scores(f"ensembles.{arch}", split)
            assert out.groupby("query_id").size().max() <= 25
        assert mrr_at_k(read_scores(f"ensembles.{arch}", "val"), truth) > 0.3

    import json

    report = json.loads((tmp_path / "models" / "ensembles.arch1_late_fusion" / "fusion.json").read_text())
    assert {"oof_val_mrr", "in_sample_val_mrr", "fold_weights"} <= set(report)
    # val export = out-of-fold scores, so it matches the reported honest number, not the in-sample one
    val = read_scores("ensembles.arch1_late_fusion", "val")
    assert mrr_at_k(val, truth) == pytest.approx(report["oof_val_mrr"])
