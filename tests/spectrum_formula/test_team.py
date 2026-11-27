"""Tests for the team-protocol adapter. Owner: Member 4."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sf_common import features as F
from sf_common import team


def test_pipeline_is_found_and_exposes_the_protocol_functions():
    cp = team.load_pipeline()
    for name in ("make_split", "select_train_rows", "build_pool", "rank_by_fingerprint", "evaluate", "FP_BITS"):
        assert hasattr(cp, name), name
    assert cp.FP_BITS == 2048


def _parquet(tmp_path, n=7, row_group_size=3):
    rng = np.random.default_rng(0)
    precursor = np.linspace(300, 600, n)
    peaks = [np.sort(rng.uniform(50, 290, 5 + i)) for i in range(n)]  # all below precursor
    table = pa.table(
        {
            "ms2_mzs": pa.array(peaks, pa.list_(pa.float64())),
            "ms2_normalized_intensities": pa.array([np.ones(len(p)) for p in peaks], pa.list_(pa.float64())),
            "collision_energy_ev": pa.array(
                [[10.0 * (i + 1)] if i % 2 else None for i in range(n)], pa.list_(pa.float64())
            ),
        }
    )
    path = tmp_path / "spectra.parquet"
    pq.write_table(table, path, row_group_size=row_group_size)
    return str(path), precursor, peaks


def test_spectra_arrays_are_aligned_with_row_order_across_row_groups(tmp_path):
    path, precursor, peaks = _parquet(tmp_path)
    order = [5, 0, 6, 2]  # unsorted request spanning several row groups
    rows = pd.DataFrame({"row_id": order, "precursor_mz": precursor[order], "adduct": ["[M+H]+"] * len(order)})
    X, meta = team.spectra_arrays(path, rows)
    assert X.shape == (4, F.N_BINS) and meta.shape == (4, F.N_META)
    for k, row in enumerate(order):
        assert int((X[k] > 0).sum()) == len({int(m) for m in peaks[row]})  # one bin per distinct integer m/z
    ce_missing = meta[:, -1].tolist()
    assert ce_missing == [0.0, 1.0, 1.0, 1.0]  # rows 5 has CE; 0, 6, 2 are even -> missing


def test_rankings_to_scores_is_a_valid_candidate_scores_table():
    from casmi.interfaces import validate_candidate_scores

    pool = pd.DataFrame({"key": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"]})
    scores = team.rankings_to_scores({"m1": [2, 0], "m2": [1]}, pool)
    validate_candidate_scores(scores)
    assert scores[scores.query_id == "m1"].sort_values("score", ascending=False).inchikey14.tolist() == [
        "CCCCCCCCCCCCCC",
        "AAAAAAAAAAAAAA",
    ]


def test_external_candidates_renames_to_the_pipeline_schema():
    pool = pd.DataFrame(
        {"inchikey14": ["AAAAAAAAAAAAAA"], "smiles": ["CC"], "formula": ["C2H6"], "monoisotopic_mass": [30.047]}
    )
    ext = team.external_candidates([pool, pool])
    assert list(ext.columns) == ["key", "smiles", "mass"] and len(ext) == 1
    assert team.external_candidates([]) is None


def test_formula_targets_handles_unknown_formulas():
    pool = pd.DataFrame({"formula": ["C2H6O", "", None]})
    t = team.formula_targets(pool)
    assert t.shape == (3, len(F.ELEMENTS)) and t[1].sum() == 0 and t[2].sum() == 0
    assert t[0, 0] == pytest.approx(np.log(3))


def test_rrf_fuse_prefers_items_ranked_high_in_both_lists():
    from sf_common import b2gate

    fused = b2gate.rrf_fuse([[5, 1, 2], [1, 7, 5]], [1.0, 1.0], top=3)
    assert fused[:2] == [1, 5] and set(fused) == {1, 5, 7}
    assert b2gate.rrf_fuse([[3, 4]], top=1) == [3]


def test_gate_module_from_member_7_is_importable_and_exposes_the_gate():
    from sf_common import b2gate

    m7 = b2gate.load_m7()
    for name in ("b2_top_scores", "gate", "tune_gate"):
        assert hasattr(m7, name), name
    assert m7.gate({"a": [1]}, {"a": [2]}, {"a": 0.9}, 0.5) == {"a": [2]}
    assert m7.gate({"a": [1]}, {"a": [2]}, {"a": 0.1}, 0.5) == {"a": [1]}
