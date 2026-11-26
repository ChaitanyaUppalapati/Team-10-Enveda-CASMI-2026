"""v3 (PubChem tier recall, priors, hidden-like tuning) end to end on synthetic files."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("rdkit")
import test_teamsplit as tt  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

cp, T = tt.cp, tt.T
from ensemble_common.standalone import recall as R  # noqa: E402
from ensemble_common.standalone.candidates import pool_fingerprints  # noqa: E402

DECOYS = ["CCCCCO", "CC(C)CCO", "CCOCCC", "CCCC(C)O", "OCc1ccccc1", "Cc1ccc(O)cc1", "CCN(C)C", "COC(C)=O",
          "CC(C)C(=O)O", "OC(=O)CCCC", "NCCC(=O)O", "CC(N)CC(=O)O"]


def write_tier(folder: Path, smiles):
    """pc_* arrays in the tier format: mass-sorted SMILES bytes with offsets and aligned priors."""
    rows = sorted((Descriptors.ExactMolWt(Chem.MolFromSmiles(s)), s) for s in smiles)
    data = [s.encode() for _, s in rows]
    off = np.r_[0, np.cumsum([len(d) for d in data])].astype(np.int64)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "pc_mass.npy", np.array([m for m, _ in rows], dtype=np.float64))
    np.save(folder / "pc_off.npy", off)
    np.save(folder / "pc_smiles.npy", np.frombuffer(b"".join(data), dtype=np.uint8))
    rng = np.random.default_rng(0)
    np.save(folder.parent / "pc_lsid.npy", rng.uniform(0, 4, len(rows)).astype(np.float16))
    np.save(folder.parent / "pc_lpmid.npy", rng.uniform(0, 2, len(rows)).astype(np.float16))
    np.save(folder.parent / "pc_np.npy", (rng.random(len(rows)) < 0.3).astype(np.uint8))
    return [s for _, s in rows]


def test_tier_window_and_smiles(tmp_path):
    order = write_tier(tmp_path / "prior" / "tier", ["CCO", "c1ccccc1", "CCCO"])
    tier = R.PubChemTier.find(str(tmp_path))
    assert len(tier) == 3 and tier.smiles([0, 1, 2]) == order
    lo, hi = tier.bounds(Descriptors.ExactMolWt(Chem.MolFromSmiles("CCCO")), 10)
    assert tier.smiles(range(lo, hi)) == ["CCCO"]
    assert tier.prior([0, 2]).shape == (2,) and set(tier.is_np([0, 1, 2])) <= {0, 1}


def test_priors_scale_linearly_with_pool_size():
    """Regression: np.isin on string keys compares every pair (hours at 1.4M x 730k on Kaggle)."""
    import time

    n = 60_000
    pool = pd.DataFrame({"key": [f"K{i:013d}" for i in range(n)]})
    coconut = frozenset(f"K{i:013d}" for i in range(0, 2 * n, 2))
    t = time.time()
    _, npf = R.candidate_priors(pool, coconut, log=lambda *_: None)
    assert time.time() - t < 3.0
    assert npf.sum() == n // 2 and npf[0] == 1 and npf[1] == 0


def test_unique_leaders_moves_the_weaker_duplicate():
    out = R.unique_leaders({"a": [5, 1], "b": [5, 2], "c": [7, 3]}, {"a": 2.0, "b": 1.0, "c": 0.0})
    assert out == {"a": [5, 1], "b": [2, 5], "c": [7, 3]}


@pytest.mark.parametrize("workers", [1, 2])
def test_v3_end_to_end(tmp_path, workers):
    tt.make_files(tmp_path)
    meta = cp.load_meta(str(tmp_path))
    queries, lib_mask = cp.make_split(meta, frac=0.3, max_query_spectra=4)
    truth = cp.truth_table(queries).reset_index(drop=True)
    pool = cp.build_pool(meta, lib_mask)
    pool_fp = pool_fingerprints(pool, None, None, cp.FP_BITS, workers=1)
    key_index = dict(zip(pool.key, range(len(pool)), strict=True))
    test, test_path = cp.load_test(str(tmp_path))
    val_mass = queries.groupby("molecule_id").neutral_mass.median().reindex(truth.molecule_id)
    test_mols = test.molecule_id.drop_duplicates().tolist()
    test_mass = test.groupby("molecule_id").neutral_mass.median().reindex(test_mols)
    train_rows = cp.select_train_rows(meta, lib_mask, per_structure=4)
    libs = T.build_libraries(cp, meta, lib_mask, pool, val_mass, test_mass, tt.CFG)
    mats = T.feature_pass(str(tmp_path / "train.parquet"), len(meta), libs,
                          {"train": train_rows.row_id.to_numpy(), "val": queries.row_id.to_numpy()}, tt.CFG,
                          log=lambda *_: None)
    X_te = T.bin_file(test_path, len(test), tt.CFG)
    model = tt.NumpyFP().fit(mats["train"], T.side_features(cp, train_rows),
                             train_rows.inchikey14.map(key_index).to_numpy(), pool_fp)
    P_va = model.predict(mats["val"], T.side_features(cp, queries))
    P_te = model.predict(X_te, T.side_features(cp, test))

    # tier: every training structure (so removed answers can be found again) plus same-mass decoys
    write_tier(tmp_path / "pc" / "casmi26-pubchem-tier", tt.SMILES + DECOYS)
    tier = R.PubChemTier.find(str(tmp_path / "pc"))
    mols = truth.molecule_id.tolist() + test_mols
    masses = np.r_[val_mass.to_numpy(), test_mass.to_numpy()]
    probs = np.vstack([R.molecule_probs(queries, P_va, truth.molecule_id), R.molecule_probs(test, P_te, test_mols)])
    hits, frame, frame_fp = R.tier_shortlists(tier, mols, masses, probs, ppm=10, k_fp=3, k_mix=3,
                                              workers=workers, log=lambda *_: None)
    assert set(hits) == set(mols) and len(frame) and frame.key.str.len().eq(14).all()
    pool_all, fp_all, mol_idx = R.extend_pool(pool, pool_fp, tier, hits, frame, frame_fp)
    n_main = len(pool)
    assert pool_all.key.iloc[:n_main].equals(pool.key) and (pool_all.tier_row.iloc[:n_main] == -1).all()
    assert len(fp_all) == len(pool_all) and pool_all.key.is_unique
    pop, npf = R.candidate_priors(pool_all, tier=tier, tier_rows=pool_all.tier_row.to_numpy(), log=lambda *_: None)
    assert pop.shape == (len(pool_all),) and npf.dtype == np.uint8

    tk = dict(zip(truth.molecule_id, truth.true_key, strict=True))
    args = (cp, queries, mats["val"], P_va, libs.val, pool_all, fp_all, pop, npf, n_main, tt.CFG)
    sc_std = R.score_v3(*args, mols=truth.molecule_id.tolist(), extra_idx=mol_idx, truth_keys=tk)
    sc_nov = R.score_v3(*args, mols=truth.molecule_id.tolist(), extra_idx=mol_idx, truth_keys=tk, novel=True)
    assert set(sc_std.src) == set(R.SOURCES)
    # novel copies: the truth never keeps a library score, and it is found only through the tier shortlist
    has = sc_nov.truth >= 0
    assert has.mean() > 0.5 and (sc_std.truth >= 0).all()
    assert (sc_nov.src["library"][sc_nov.truth[has]] == 0).all()
    shortlisted = {pool_all.key.to_numpy()[j] for idx in mol_idx.values() for j in idx}
    assert all(tk[m] in shortlisted for m, h in zip(sc_nov.mols, has, strict=True) if h)

    rep = R.tune_v3(cp, sc_std, sc_nov, truth, pool_all, n_main, tt.CFG, log=lambda *_: None)
    s = rep["summary"]
    assert {"v2_global_z", "v3_gated_z", "pop", "np"} <= set(s.method) and rep["best"] == s.method[0]
    assert s.objective.between(0, 1).all() and s["novel all"].between(0, 1).all()
    # the removed truth is never used as filler in the novel rankings
    f_nov = R.apply_v3(rep, sc_nov, tt.CFG, "fp_tani")
    ranks = R.rankings_v3(cp, sc_nov, f_nov, pool_all, n_main, rep["exclude"])
    keys = pool_all.key.to_numpy()
    for m, r in ranks.items():
        assert len(r) == 25 and len(set(r)) == 25
        if m in rep["exclude"] and not (sc_nov.truth[sc_nov.mols.index(m)] >= 0):
            assert rep["exclude"][m] not in keys[r]

    sc_te = R.score_v3(cp, test, X_te, P_te, libs.test, pool_all, fp_all, pop, npf, n_main, tt.CFG,
                       mols=test_mols, extra_idx=mol_idx)
    f_te = R.apply_v3(rep, sc_te, tt.CFG)
    ranks_te = R.rankings_v3(cp, sc_te, f_te, pool_all, n_main)
    sub = cp.write_submission(ranks_te, pool_all, test, path=str(tmp_path / "submission.csv"))
    assert len(sub) == len(test_mols) and (sub.smiles.str.count(";") == 24).all()
    assert R.describe(rep["params"][rep["best"]])


def test_no_tier_falls_back_cleanly():
    pool = pd.DataFrame({"key": ["A" * 14], "smiles": ["CCO"], "formula": ["C2H6O"], "mass": [46.0], "n_ref": [1]})
    hits, frame, fp = R.tier_shortlists(None, ["m"], np.array([46.0]), np.zeros((1, 16)), log=lambda *_: None)
    pool_all, fp_all, mol_idx = R.extend_pool(pool, np.zeros((1, 2), np.uint8), None, hits, frame, fp)
    assert hits == {} and mol_idx == {} and len(pool_all) == 1 and (pool_all.tier_row == -1).all()
