"""Standalone pipeline on synthetic spectra: library leave-query-out, validation sets, test prediction."""

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("rdkit")
pytest.importorskip("sklearn")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from ensemble_common.standalone import pipeline as P  # noqa: E402
from ensemble_common.standalone.chem import adduct_delta, exact_mass, morgan_bits, neutral_mass  # noqa: E402
from ensemble_common.standalone.library import CentroidLibrary  # noqa: E402
from ensemble_common.standalone.spectra import bin_one, pool  # noqa: E402
from rdkit import Chem  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

SMILES = [
    "CCO", "CCCO", "CCCCO", "CC(C)O", "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",
    "Oc1ccccc1", "Nc1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1", "CC(=O)OC", "CCOC(C)=O", "CCN", "CCCN", "CCCCN",
    "OCC(O)CO", "CC(O)C(=O)O", "OC(=O)CC(O)(CC(O)=O)C(O)=O", "c1ccc2ccccc2c1", "CC(C)(C)O", "CCOCC", "CC(C)=O",
    "CCC(C)=O", "O=Cc1ccccc1", "OC(=O)c1ccccc1", "Clc1ccccc1", "Brc1ccccc1", "CCCCCC", "CCCCCCC", "CCCCCCCC",
    "OCCCCO", "NCCN", "NC(=O)C", "CC(N)C(=O)O", "NCC(=O)O", "CC(C)CC(N)C(=O)O", "OC(=O)CCC(N)C(=O)O",
    "c1ccc(cc1)C(=O)OC", "COc1ccccc1", "CC(C)c1ccccc1", "Cc1ccc(C)cc1", "Cc1cccc(C)c1", "OC(=O)C=CC(O)=O",
    "CCCCCCCCO", "CCCCCCCCCO", "c1ccc(O)c(O)c1", "Oc1ccc(O)cc1", "CC1=CC(=O)CC(C)(C)C1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
]


def spectrum(key: str, rng, drop=0.25):
    base = np.random.default_rng(zlib.crc32(key.encode()))
    mz = np.sort(base.uniform(30, 400, 14))
    keep = rng.random(14) > drop
    return list(mz[keep] + rng.normal(0, 0.01, keep.sum())), list(base.uniform(0.2, 1, 14)[keep])


def make_files(tmp_path, n_per=5):
    rng = np.random.default_rng(0)
    rows, keys = [], {}
    for smi in SMILES:
        key = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))[:14]
        keys[smi] = key
        mass = exact_mass(smi)
        for _ in range(n_per):
            mz, it = spectrum(key, rng)
            rows.append({"normalized_smiles": smi, "inchikey14": key, "ms2_mzs": mz, "ms2_normalized_intensities": it,
                         "precursor_mz": mass + adduct_delta("[M+H]+"), "adduct": "[M+H]+"})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet")
    trows = []
    for j, smi in enumerate(SMILES[:6]):
        for s in range(2):
            mz, it = spectrum(keys[smi], rng)
            trows.append({"molecule_id": f"mol{j}", "spectrum_id": f"s{j}_{s}", "ms2_mzs": mz,
                          "ms2_normalized_intensities": it, "precursor_mz": exact_mass(smi) + 1.007276,
                          "adduct": "[M+H]+"})
    pq.write_table(pa.Table.from_pylist(trows), tmp_path / "test.parquet")
    return keys


class RidgeFP:
    """Stand-in for the Kaggle neural net: any object with fit/predict works."""

    def fit(self, X, Y):
        self.m = Ridge(alpha=1.0).fit(X.astype(np.float32), Y.astype(np.float32))
        return self

    def predict(self, X):
        return np.clip(self.m.predict(X), 0, 1).astype(np.float32)


def test_chem_helpers():
    assert neutral_mass(100.0 + 1.007276, "[M+H]+") == pytest.approx(100.0)
    assert neutral_mass(100.0 - 1.007276, "[M-H]-") == pytest.approx(100.0)
    assert adduct_delta(None) == adduct_delta("[M+H]+") and adduct_delta("weird-") == adduct_delta("[M-H]-")
    assert morgan_bits("CCO").sum() > 0 and morgan_bits("not a smiles").sum() == 0


def test_library_leave_query_out():
    lib = CentroidLibrary(2, 3)
    q = np.array([1, 0, 0], dtype=np.float32)
    other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    lib.add(0, q), lib.add(0, other), lib.add(1, other)
    with_self = lib.scores(q, np.array([0, 1]))
    without = lib.scores(q, np.array([0, 1]), subtract=(0, q, 1))
    assert with_self[0] > 0.7 and without[0] == pytest.approx(0.0)
    assert lib.scores(q, np.array([0, 1]), absent=np.array([True, False]))[0] == 0.0


def test_pool_and_bin_are_unit_norm():
    v = bin_one([100.0, 200.0], [1.0, 4.0], 500.0, 1.0)
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert np.linalg.norm(pool(np.stack([v, v * 0.5]))) == pytest.approx(1.0)


def test_pipeline_end_to_end(tmp_path):
    make_files(tmp_path)
    cfg = P.Config(train_path=str(tmp_path / "train.parquet"), test_path=str(tmp_path / "test.parquet"),
                   heldout_permille=200, covered_permille=400, max_fp_train=500, fp_per_structure=3, n_random=20,
                   ppm=5.0, batch_size=50)
    st = P.build_state(cfg, log=lambda *_: None)
    assert len(st.absent_q) >= 3 and len(st.covered_q) >= 3
    # reserved validation spectra must not appear in the fingerprint training set
    assert len(st.fp_X) > 0 and len(st.fp_X) == len(st.fp_Y)
    fp = RidgeFP().fit(st.fp_X, st.fp_Y)
    rep = P.tune_and_report(st, cfg, fp, log=lambda *_: None)
    assert sum(rep["weights"].values()) == pytest.approx(1.0)
    # reference-covered queries are findable with the library even though they cannot see themselves
    assert rep["per_source"]["covered"]["library"] > 0.5
    # reference-absent: the held-out truth must get exactly zero library evidence (no leakage)
    absent = P.score_queries(st.absent_q, st, cfg, fp, library_absent=True, log=lambda *_: None)["library"]
    truth_rows = absent.merge(P.truth_frame(st.absent_q, st), on=["query_id", "inchikey14"])
    assert len(truth_rows) > 0 and (truth_rows["score"] == 0).all()
    assert rep["fused"]["covered"] >= rep["per_source"]["covered"]["fingerprint"] - 1e-9
    out = P.predict_test(st, cfg, fp, rep["weights"], log=lambda *_: None)
    assert out.groupby("query_id").size().max() <= 25 and out.query_id.nunique() == 6
    final = P.attach_smiles(out, st)
    assert final.smiles.notna().all()
