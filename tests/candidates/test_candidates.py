"""Tests for the candidate pool, adduct-aware lookup and coverage audit. Owner: Member 4."""

import pandas as pd
import pytest

from casmi.candidates.coconut import load_coconut
from casmi.candidates.coverage import coverage, per_query_hit, pool_membership
from casmi.candidates.lookup import ADDUCT_DELTAS, candidates_for_queries, neutral_mass
from casmi.candidates.pool import build_pool, find_smiles_column
from casmi.candidates.standardize import POOL_COLUMNS, standardize_many, standardize_smiles

CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CAFFEINE_KEY = "RYYVLZVUVIJVGH"
CAFFEINE_MASS = 194.080376
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_KEY = "BSYNRYMUTXBXSQ"
GLUCOSE = "OCC1OC(O)C(O)C(O)C1O"


def _pool() -> pd.DataFrame:
    return build_pool_from_smiles([CAFFEINE, ASPIRIN, GLUCOSE])


def build_pool_from_smiles(smiles):
    return standardize_many(smiles, mass_range=(100.0, 1500.0))


# ---------- standardisation ----------
def test_standardize_known_molecule():
    key, smiles, formula, mass = standardize_smiles(CAFFEINE)
    assert key == CAFFEINE_KEY
    assert formula == "C8H10N4O2"
    assert mass == pytest.approx(CAFFEINE_MASS, abs=1e-5)


def test_standardize_keeps_largest_fragment_of_a_salt():
    sodium_salt = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"
    assert standardize_smiles(sodium_salt)[0] == standardize_smiles("CC(=O)Oc1ccccc1C(=O)[O-]")[0]


@pytest.mark.parametrize("bad", ["not a smiles", "", "C"])  # unparsable, empty, below mass range
def test_standardize_rejects_unusable_input(bad):
    assert standardize_smiles(bad) is None


def test_standardize_many_dedupes_and_has_pool_columns():
    pool = build_pool_from_smiles([CAFFEINE, CAFFEINE, "bogus", None])
    assert list(pool.columns) == list(POOL_COLUMNS)
    assert pool["inchikey14"].tolist() == [CAFFEINE_KEY]


def test_parallel_standardisation_matches_serial():
    smiles = [CAFFEINE, ASPIRIN, GLUCOSE, "bogus"] * 3
    serial = standardize_many(smiles, workers=1, chunk_size=2)
    parallel = standardize_many(smiles, workers=2, chunk_size=2)
    pd.testing.assert_frame_equal(serial, parallel)


# ---------- adducts ----------
def test_neutral_mass_protonated():
    assert neutral_mass(CAFFEINE_MASS + 1.007276, "[M+H]+") == pytest.approx(CAFFEINE_MASS, abs=1e-4)


def test_neutral_mass_deprotonated_and_unknown():
    assert neutral_mass(CAFFEINE_MASS - 1.007276, "[M-H]-") == pytest.approx(CAFFEINE_MASS, abs=1e-4)
    assert pd.isna(neutral_mass(300.0, "[M+Mystery]+"))
    assert pd.isna(neutral_mass(300.0, None))


def test_all_test_set_adducts_are_defined():
    assert len(ADDUCT_DELTAS) == 10


@pytest.mark.parametrize("adduct", sorted(ADDUCT_DELTAS))
def test_adduct_round_trip(adduct):
    mz = CAFFEINE_MASS + ADDUCT_DELTAS[adduct]
    assert neutral_mass(mz, adduct) == pytest.approx(CAFFEINE_MASS, abs=1e-9)


# ---------- lookup ----------
def _spectra(rows):
    return pd.DataFrame(rows, columns=["query_id", "precursor_mz", "adduct"])


def test_lookup_finds_mass_match_only():
    spectra = _spectra([("q1", CAFFEINE_MASS + 1.007276, "[M+H]+")])
    hits = candidates_for_queries(spectra, _pool(), ppm=10)
    assert hits["inchikey14"].tolist() == [CAFFEINE_KEY]
    assert hits["ppm_error"].iloc[0] < 1


def test_lookup_uses_the_adduct():
    # The same m/z read as [M+Na]+ implies a different neutral mass, so caffeine must not match.
    mz = CAFFEINE_MASS + 1.007276
    hits = candidates_for_queries(_spectra([("q1", mz, "[M+Na]+")]), _pool(), ppm=10)
    assert CAFFEINE_KEY not in hits["inchikey14"].tolist()


def test_lookup_ppm_window_is_respected():
    mz = CAFFEINE_MASS + 1.007276 + 0.01  # about 51 ppm away
    spectra = _spectra([("q1", mz, "[M+H]+")])
    assert candidates_for_queries(spectra, _pool(), ppm=10).empty
    assert len(candidates_for_queries(spectra, _pool(), ppm=100)) == 1


def test_lookup_unions_spectra_of_one_query_and_dedupes():
    spectra = _spectra(
        [
            ("q1", CAFFEINE_MASS + 1.007276, "[M+H]+"),
            ("q1", CAFFEINE_MASS - 1.007276, "[M-H]-"),  # same molecule, other polarity
            ("q1", 180.042259 + 1.007276, "[M+H]+"),  # aspirin
        ]
    )
    hits = candidates_for_queries(spectra, _pool(), ppm=10)
    assert sorted(hits["inchikey14"]) == sorted([CAFFEINE_KEY, ASPIRIN_KEY])
    assert not hits.duplicated(["query_id", "inchikey14"]).any()


def test_lookup_skips_unknown_adducts_and_handles_no_hits():
    hits = candidates_for_queries(_spectra([("q1", 195.0877, "[M+Mystery]+")]), _pool())
    assert hits.empty
    assert {"query_id", "inchikey14", "smiles", "ppm_error"} <= set(hits.columns)


def test_lookup_rejects_nonpositive_ppm():
    with pytest.raises(ValueError):
        candidates_for_queries(_spectra([("q1", 195.0, "[M+H]+")]), _pool(), ppm=0)


# ---------- coverage ----------
def test_coverage_counts_queries_with_true_structure():
    cands = pd.DataFrame({"query_id": ["q1", "q1", "q2"], "inchikey14": [CAFFEINE_KEY, ASPIRIN_KEY, ASPIRIN_KEY]})
    truth = pd.DataFrame({"query_id": ["q1", "q2", "q3"], "inchikey14": [CAFFEINE_KEY, CAFFEINE_KEY, ASPIRIN_KEY]})
    assert per_query_hit(cands, truth).tolist() == [True, False, False]
    assert coverage(cands, truth) == pytest.approx(1 / 3)


def test_coverage_empty_truth_and_duplicate_queries():
    empty = pd.DataFrame({"query_id": [], "inchikey14": []})
    assert coverage(empty, empty) == 0.0
    dup = pd.DataFrame({"query_id": ["q1", "q1"], "inchikey14": [CAFFEINE_KEY, ASPIRIN_KEY]})
    with pytest.raises(ValueError):
        coverage(empty, dup)


def test_pool_membership_ignores_mass_filter():
    truth = pd.DataFrame({"query_id": ["a", "b"], "inchikey14": [CAFFEINE_KEY, "ZZZZZZZZZZZZZZ"]})
    assert pool_membership(_pool(), truth) == 0.5


# ---------- pool IO ----------
def test_find_smiles_column_prefers_known_names():
    assert find_smiles_column(["id", "Canonical_SMILES", "name"]) == "Canonical_SMILES"
    assert find_smiles_column(["id", "my_smiles_col"]) == "my_smiles_col"
    with pytest.raises(ValueError):
        find_smiles_column(["id", "name"])


def test_build_and_load_round_trip(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"id": [1, 2, 3], "canonical_smiles": [CAFFEINE, ASPIRIN, "bogus"]}).to_csv(raw, index=False)
    pool = build_pool(raw)
    assert len(pool) == 2
    pool.to_parquet(tmp_path / "coconut.parquet", index=False)
    assert load_coconut(tmp_path)["inchikey14"].tolist() == pool["inchikey14"].tolist()


def test_load_coconut_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_dataset"):
        load_coconut(tmp_path)
