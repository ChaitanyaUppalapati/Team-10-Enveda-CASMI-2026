"""Team-split late fusion on synthetic parquet files, using Member 3's real casmi_pipeline module."""

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "baselines" / "b2_library_search"))

pytest.importorskip("rdkit")
import casmi_pipeline as cp  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from ensemble_common.standalone import teamsplit as T  # noqa: E402
from ensemble_common.standalone.candidates import pool_fingerprints  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors, rdMolDescriptors  # noqa: E402

SMILES = [
    "CCO", "CCCO", "CCCCO", "CC(C)O", "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",
    "Oc1ccccc1", "Nc1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1", "CC(=O)OC", "CCOC(C)=O", "CCN", "CCCN", "CCCCN",
    "OCC(O)CO", "CC(O)C(=O)O", "OC(=O)CC(O)(CC(O)=O)C(O)=O", "c1ccc2ccccc2c1", "CC(C)(C)O", "CCOCC", "CC(C)=O",
    "CCC(C)=O", "O=Cc1ccccc1", "OC(=O)c1ccccc1", "Clc1ccccc1", "Brc1ccccc1", "CCCCCC", "CCCCCCC", "CCCCCCCC",
    "OCCCCO", "NCCN", "NC(=O)C", "CC(N)C(=O)O", "NCC(=O)O", "CC(C)CC(N)C(=O)O", "OC(=O)CCC(N)C(=O)O",
    "c1ccc(cc1)C(=O)OC", "COc1ccccc1", "CC(C)c1ccccc1", "Cc1ccc(C)cc1", "Cc1cccc(C)c1", "OC(=O)C=CC(O)=O",
    "CCCCCCCCO", "CCCCCCCCCO", "c1ccc(O)c(O)c1", "Oc1ccc(O)cc1", "CC1=CC(=O)CC(C)(C)C1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
]
CFG = T.FusionConfig(n_bins=400, chunk_rows=37, n_random=30, min_bin=3, ppm=10.0)


def spectrum(key, mass, rng, drop=0.25):
    base = np.random.default_rng(zlib.crc32(key.encode()))
    mz = np.sort(base.uniform(15, mass - 1, 14))
    keep = rng.random(14) > drop
    return list(mz[keep] + rng.normal(0, 0.005, keep.sum())), list(base.uniform(0.2, 1, 14)[keep])


def make_files(tmp_path):
    rng = np.random.default_rng(0)
    rows, info = [], {}
    for j, smi in enumerate(SMILES):
        mol = Chem.MolFromSmiles(smi)
        key, mass = Chem.MolToInchiKey(mol)[:14], Descriptors.ExactMolWt(mol)
        info[smi] = (key, mass)
        for s in range(6):
            lib = "enveda-np-examples" if (j % 9 == 0 and s < 3) else ("libA" if s % 2 else "libB")
            mz, it = spectrum(key, mass, rng)
            mode = "positive"
            rows.append({"ingest_lib": lib, "inchikey14": key, "normalized_smiles": smi,
                         "molecular_formula": rdMolDescriptors.CalcMolFormula(mol), "ionization_mode": mode,
                         "instrument_type": "QTOF", "adduct": "[M+H]+", "precursor_mz": mass + cp.PROTON,
                         "precursor_error_ppm": 0.0, "num_peaks": len(mz), "ms2_mzs": mz,
                         "ms2_normalized_intensities": it})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet", row_group_size=50)
    trows = []
    for j, smi in enumerate(SMILES[:8]):
        key, mass = info[smi]
        for s in range(2):
            mz, it = spectrum(key, mass, rng)
            trows.append({"molecule_id": f"m{j}", "spectrum_id": f"s{j}_{s}", "adduct": "[M+H]+",
                          "ionization_mode": "positive", "precursor_mz": mass + cp.PROTON, "ms2_mzs": mz,
                          "ms2_normalized_intensities": it})
    pq.write_table(pa.Table.from_pylist(trows), tmp_path / "test.parquet")


class NumpyFP:
    """Stand-in for the Kaggle network: ridge regression from [X, S] to fingerprint bits."""

    def fit(self, X, S, target_idx, fp_packed, val=None, log=print):
        A = np.hstack([X.astype(np.float64), S])
        Y = np.unpackbits(fp_packed[target_idx], axis=1).astype(np.float64)
        self.W = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ Y)
        return self

    def predict(self, X, S):
        return np.clip(np.hstack([X.astype(np.float64), S]) @ self.W, 0, 1).astype(np.float32)


def test_bin_batch_drops_precursor_and_is_unit_norm():
    t = pa.table({"ms2_mzs": [[50.0, 99.8, 120.0], [10.0, 20.0]], "ms2_normalized_intensities": [[1.0, 1.0, 1.0],
                                                                                                    [0.5, 1.0]],
                  "precursor_mz": [100.0, 5.0]}).to_batches()[0]
    X = T.bin_batch(t, np.array([0, 1]), 200, 1.0)
    assert X[0, 50] == pytest.approx(1.0) and X[0, 99] == 0 and X[0, 120] == 0
    # every peak is above the precursor: keep them rather than return an empty spectrum
    assert np.linalg.norm(X[1]) == pytest.approx(1.0)


def test_rrf_ties_and_reciprocal_ranks():
    seg = np.array([0, 0, 0, 1, 1])
    out = T.normalize(np.array([3.0, 3.0, 1.0, 0.0, 2.0]), seg, 2, "rrf", k=0)
    assert out.tolist() == [1.0, 1.0, 1 / 3, 1 / 2, 1.0]
    sc = T.Scores(mols=["a", "b"], mass=np.zeros(2), seg=seg, cand=np.array([5, 2, 9, 1, 3]), truth=np.array([0, 3]))
    rr = T.reciprocal_ranks(np.array([3.0, 3.0, 1.0, 0.0, 2.0]), sc)
    assert rr.tolist() == [0.5, 0.5]  # tie with a lower pool index ranks first


def test_end_to_end_on_team_split(tmp_path):
    make_files(tmp_path)
    meta = cp.load_meta(str(tmp_path))
    queries, lib_mask = cp.make_split(meta, frac=0.3, max_query_spectra=4)
    truth = cp.truth_table(queries).reset_index(drop=True)
    assert len(truth) >= 6 and {1, 2} <= set(truth.cls)
    pool = cp.build_pool(meta, lib_mask)
    pool_fp = pool_fingerprints(pool, None, None, cp.FP_BITS, workers=1)
    key_index = dict(zip(pool.key, range(len(pool)), strict=True))
    test, test_path = cp.load_test(str(tmp_path))
    val_mass = queries.groupby("molecule_id").neutral_mass.median().reindex(truth.molecule_id)
    test_mass = test.groupby("molecule_id").neutral_mass.median()
    train_rows = cp.select_train_rows(meta, lib_mask, per_structure=4)
    libs = T.build_libraries(cp, meta, lib_mask, pool, val_mass, test_mass, CFG)
    mats = T.feature_pass(str(tmp_path / "train.parquet"), len(meta), libs,
                          {"train": train_rows.row_id.to_numpy(), "val": queries.row_id.to_numpy()}, CFG,
                          log=lambda *_: None)
    assert np.allclose(np.linalg.norm(mats["val"].astype(np.float32), axis=1), 1, atol=1e-2)
    # validation query rows never reach the validation library
    assert not (libs.val_cid[queries.row_id.to_numpy()] >= 0).any()
    X_te = T.bin_file(test_path, len(test), CFG)
    model = NumpyFP().fit(mats["train"], T.side_features(cp, train_rows),
                          train_rows.inchikey14.map(key_index).to_numpy(), pool_fp)
    p_va = model.predict(mats["val"], T.side_features(cp, queries))
    sc = T.score_molecules(cp, queries, mats["val"], p_va, libs.val, pool, pool_fp, CFG,
                           mols=truth.molecule_id.tolist(), truth_keys=dict(zip(truth.molecule_id, truth.true_key,
                                                                                 strict=True)))
    assert (sc.truth >= 0).all()
    # class 2 truths have no reference spectra in the validation library: zero library score
    cls2 = np.flatnonzero(truth.set_index("molecule_id").loc[sc.mols].cls.to_numpy() == 2)
    assert (sc.src["library"][sc.truth[cls2]] == 0).all()
    rep = T.tune_and_evaluate(cp, sc, truth, pool, CFG, log=lambda *_: None)
    s = rep["summary"]
    assert set(T.SOURCES) <= set(s.method) and {"global_z", "gated_rrf"} <= set(s.method)
    assert s.balanced.between(0, 1).all() and rep["best"] == s.method[0]
    # class 1 truths keep references from other libraries, so the library source finds them
    assert rep["tables"]["library"].loc["class 1", "MRR@25"] > 0.5
    sc_te = T.score_molecules(cp, test, X_te, model.predict(X_te, T.side_features(cp, test)), libs.test, pool,
                              pool_fp, CFG)
    ranks = T.rankings(cp, sc_te, T.apply_best(rep, sc_te, CFG), pool)
    sub = cp.write_submission(ranks, pool, test, path=str(tmp_path / "submission.csv"))
    assert len(sub) == test.molecule_id.nunique() and (sub.smiles.str.count(";") < 25).all()
    # test library uses every training spectrum, so test structures (all in training) rank first
    top1 = [pool.key.to_numpy()[ranks[m][0]] for m in sc_te.mols]
    want = [Chem.MolToInchiKey(Chem.MolFromSmiles(SMILES[int(m[1:])]))[:14] for m in sc_te.mols]
    assert np.mean([a == b for a, b in zip(top1, want, strict=True)]) >= 0.75
    table = T.score_table(sc_te, pool, T.apply_best(rep, sc_te, CFG))
    assert {"query_id", "inchikey14", "score", *T.SOURCES} <= set(table.columns)
    assert T.coverage(sc, truth)["truth_in_window"] == 1.0
    T.describe(rep["params"]["gated_z"])
