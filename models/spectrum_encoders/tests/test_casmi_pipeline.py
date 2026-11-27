"""Member 3 checks for common/casmi_pipeline.py (shared by B1, B2 and the spectrum encoders).

Run with: pytest models/spectrum_encoders/tests -q
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPELINE = HERE.parent / "common" / "casmi_pipeline.py"

# Load by file path under a unique name: other members also have a top-level `common` package.
_spec = importlib.util.spec_from_file_location("member3_casmi_pipeline", PIPELINE)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)

NOTEBOOKS = [
    "baselines/b1_mass_only/notebooks/01_mass_only_baseline.ipynb",
    "baselines/b2_library_search/notebooks/02_library_search_baseline.ipynb",
    "models/spectrum_encoders/arch1_residual_mlp/notebooks/03_residual_mlp.ipynb",
    "models/spectrum_encoders/arch2_peakset_transformer/notebooks/04_peak_set_transformer.ipynb",
]


@pytest.mark.parametrize("rel", NOTEBOOKS)
def test_notebook_embeds_the_shared_pipeline(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} is not in this checkout")
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    embedded = ["".join(c["source"]) for c in cells if "".join(c["source"]).startswith("%%writefile casmi_pipeline.py")]
    assert len(embedded) == 1, "expected exactly one %%writefile casmi_pipeline.py cell"
    shared = PIPELINE.read_text(encoding="utf-8")
    assert embedded[0].split("\n", 1)[1] == shared, "notebook copy differs from common/casmi_pipeline.py"


def test_neutral_mass_inverts_adducts():
    mass = 300.123
    assert pipeline.neutral_mass(mass + pipeline.PROTON, "[M+H]+") == pytest.approx(mass)
    assert pipeline.neutral_mass(mass - pipeline.PROTON, "[M-H]-") == pytest.approx(mass)
    assert pipeline.neutral_mass((2 * mass + pipeline.PROTON), "[2M+H]+") == pytest.approx(mass)
    assert pipeline.neutral_mass((mass + 2 * pipeline.PROTON) / 2, "[M+2H]2+") == pytest.approx(mass)
    # Unknown adducts fall back to one proton in the given ionization mode.
    assert pipeline.neutral_mass(mass - pipeline.PROTON, "weird", "negative") == pytest.approx(mass)


def test_formula_mass():
    assert pipeline.formula_mass("H2O") == pytest.approx(18.0105647, abs=1e-6)
    assert pipeline.formula_mass("C6H12O6") == pytest.approx(180.0633881, abs=1e-6)
    assert np.isnan(pipeline.formula_mass("Xx2"))


def test_window_and_fill_to_k():
    pool_mass = np.linspace(100.0, 200.0, 1001)
    idx = pipeline.window_indices(pool_mass, 150.0, ppm=10)
    assert (np.abs(pool_mass[idx] - 150.0) <= 150.0 * 10e-6).all()
    ranked = pipeline.fill_to_k([500, 500, 3], pool_mass, 150.0)
    assert len(ranked) == pipeline.TOP_K
    assert len(set(ranked)) == pipeline.TOP_K
    assert ranked[:2] == [500, 3]  # given order kept, duplicates dropped, then nearest masses


def test_rank_b1_orders_by_ppm_then_prior():
    pool = pd.DataFrame({"mass": [99.0, 100.0, 100.0005, 101.0] + [200.0 + i for i in range(30)], "n_ref": 0})
    pool.loc[1, "n_ref"], pool.loc[2, "n_ref"] = 1, 5
    by_ppm = pipeline.rank_b1(pool, 100.0, ppm=10, order="ppm_then_prior")
    by_prior = pipeline.rank_b1(pool, 100.0, ppm=10, order="prior_then_ppm")
    assert by_ppm[:2] == [1, 2] and by_prior[:2] == [2, 1]
    assert len(by_ppm) == pipeline.TOP_K


def test_evaluate_mrr_and_balanced():
    keys = np.array([f"KEY{i:011d}"[:14] for i in range(30)])
    truth = pd.DataFrame(
        {
            "molecule_id": ["a", "b", "c", "d"],
            "true_key": [keys[0], keys[1], keys[2], "MISSINGKEYXXXX"],
            "cls": [1, 1, 2, 2],
            "query_lib": ["x", "enveda-np-examples", "x", "x"],
        }
    )
    order = list(range(30))
    rankings = {"a": order, "b": [5, 1] + order, "c": [9, 8, 7, 2], "d": order}
    table = pipeline.evaluate(rankings, truth, keys)
    assert table.loc["class 1", "MRR@25"] == pytest.approx((1 + 1 / 2) / 2, abs=1e-4)
    assert table.loc["class 2", "MRR@25"] == pytest.approx((1 / 4 + 0) / 2, abs=1e-4)
    assert table.loc["np proxy", "MRR@25"] == pytest.approx(0.5, abs=1e-4)
    expected = (table.loc["class 1", "MRR@25"] + table.loc["class 2", "MRR@25"]) / 2
    assert table.loc["balanced", "MRR@25"] == pytest.approx(expected, abs=1e-4)


def test_rank_beyond_top_25_scores_zero():
    keys = np.array([f"K{i:013d}" for i in range(40)])
    truth = pd.DataFrame({"molecule_id": ["a"], "true_key": [keys[30]], "cls": [1], "query_lib": ["x"]})
    ranks = pipeline.rank_table({"a": list(range(40))}, truth, keys)
    assert ranks.loc[0, "rank"] == 0 and ranks.loc[0, "rr"] == 0.0


def test_match_score_identical_and_modified():
    mz = np.array([50.0, 80.0, 120.0, 150.0])
    inten = np.array([0.5, 0.5, 0.5, 0.5])  # unit norm
    assert pipeline.match_score(mz, inten, mz, inten) == pytest.approx(1.0)
    shifted = mz + 14.0
    assert pipeline.match_score(mz, inten, shifted, inten) == 0.0
    assert pipeline.match_score(mz, inten, shifted, inten, shift=-14.0) == pytest.approx(1.0)


def _toy_meta(n_keys=200, per_key=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_keys):
        key = "".join(chr(65 + int(c)) for c in rng.integers(0, 26, 14))
        for s in range(per_key):
            rows.append(
                {
                    "ingest_lib": ["libA", "libB", "enveda-180"][s % 3],
                    "inchikey14": key,
                    "normalized_smiles": "C" * (k % 7 + 1),
                    "molecular_formula": f"C{k % 40 + 5}H{k % 30 + 6}O{k % 5 + 1}",
                    "ionization_mode": "positive",
                    "instrument_type": "Orbitrap",
                    "adduct": "[M+H]+",
                    "precursor_mz": 200.0 + k,
                    "precursor_error_ppm": 1.0,
                    "num_peaks": 10,
                }
            )
    meta = pd.DataFrame(rows)
    meta["row_id"] = np.arange(len(meta))
    for col in ["ingest_lib", "ionization_mode", "instrument_type", "adduct"]:
        meta[col] = meta[col].astype("category")
    meta["good_label"] = True
    return meta


def test_make_split_is_deterministic_and_leak_free():
    meta = _toy_meta()
    queries, library = pipeline.make_split(meta)
    queries2, library2 = pipeline.make_split(meta)
    pd.testing.assert_frame_equal(queries, queries2)
    assert (library == library2).all()
    assert len(queries) > 0
    # Query spectra are never in the library.
    assert not library[queries.row_id.to_numpy()].any()
    for key, group in queries.groupby("true_key"):
        rows = meta[meta.inchikey14 == key]
        cls = group.cls.iloc[0]
        if cls == 2:  # class 2: no reference spectra left at all
            assert not library[rows.row_id.to_numpy()].any()
        else:  # class 1: references only from libraries other than the query's
            kept = rows[library[rows.row_id.to_numpy()]]
            assert len(kept) > 0
            assert (kept.ingest_lib.astype(str) != group.query_lib.iloc[0]).all()
    # 1..max_query_spectra query spectra per held-out structure
    counts = queries.groupby("molecule_id").size()
    assert counts.between(1, 6).all()


def _tie_case():
    """Pool where tied candidates sit in mass order opposite to their inchikey14 order."""
    n = 40
    keys = [f"{chr(90 - i % 26)}{chr(65 + i // 26)}AAAAAAAAAAAA" for i in range(n)]  # descending with mass
    pool = pd.DataFrame({"key": keys, "smiles": ["C"] * n, "mass": np.linspace(300.0, 300.001, n)})
    fp = np.zeros((n, 2048), dtype=bool)
    fp[:, :8] = True  # every candidate has the same fingerprint -> identical Tanimoto
    fp[5, 8:40] = True  # one clearly different candidate
    spectra = pd.DataFrame({"molecule_id": ["q1", "q1", "q2"], "neutral_mass": [300.0005, 300.0005, 400.0]})
    probs = np.zeros((3, 2048), dtype=np.float32)
    probs[:, :8] = 0.9
    return pool, np.packbits(fp, axis=1), spectra, probs


def test_fingerprint_ties_break_by_inchikey14():
    pool, fp, spectra, probs = _tie_case()
    rankings, scores = pipeline.rank_by_fingerprint(pool, fp, spectra, probs, ppm=10, return_scores=True)
    ranked = rankings["q1"]
    keys = pool.key.to_numpy()[ranked]
    s = scores["q1"]
    assert (np.diff(s) <= 0).all(), "scores must be non-increasing in ranking order"
    for value in np.unique(s):
        tied = keys[s == value]
        assert list(tied) == sorted(tied), "tied candidates must be ordered by inchikey14"


def test_exported_scores_rerank_to_the_same_order():
    """Regression: the team ranker must reproduce the notebook order, including ties and fill candidates."""
    from casmi.interfaces import rank_candidates, validate_candidate_scores

    pool, fp, spectra, probs = _tie_case()
    rankings = pipeline.rank_by_fingerprint(pool, fp, spectra, probs, ppm=10)
    table = pipeline.fingerprint_candidate_scores(pool, fp, spectra, probs, ppm=10)
    validate_candidate_scores(table, max_per_query=pipeline.TOP_K)
    reranked = rank_candidates(table)
    keys = pool.key.to_numpy()
    for mol, ranked in rankings.items():
        expected = list(keys[ranked[: pipeline.TOP_K]])
        assert reranked[reranked.query_id == mol].inchikey14.tolist() == expected
    # q2 has no candidate in its mass window: every row is a fill candidate with a negative score
    assert (table[table.query_id == "q2"].score < 0).all()


def test_write_submission_format(tmp_path):
    pool = pd.DataFrame({"smiles": [f"C{i}" for i in range(30)]})
    test = pd.DataFrame({"molecule_id": ["m1", "m1", "m2"]})
    rankings = {"m1": list(range(25)), "m2": list(range(5, 30))}
    sub = pipeline.write_submission(rankings, pool, test, tmp_path / "submission.csv")
    assert list(sub.columns) == ["molecule_id", "smiles"]
    assert sub.molecule_id.tolist() == ["m1", "m2"]
    assert (sub.smiles.str.count(";") == pipeline.TOP_K - 1).all()
