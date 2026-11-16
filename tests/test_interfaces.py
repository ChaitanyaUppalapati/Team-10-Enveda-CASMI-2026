"""Contract tests for casmi.interfaces. Owners: Member 1 + Member 5."""

import pandas as pd
import pytest

from casmi.interfaces import (
    INCHIKEY14,
    QUERY_ID,
    RANK,
    SCORE,
    TOP_K,
    ModelRunner,
    SchemaError,
    SpectrumRecord,
    rank_candidates,
    read_scores,
    scores_path,
    to_inchikey14,
    validate_candidate_scores,
    write_scores,
)

KEY_A = "BSYNRYMUTXBXSQ"  # aspirin first block
KEY_B = "RZVAJINKPMORJF"  # paracetamol first block
KEY_C = "RYYVLZVUVIJVGH"  # caffeine first block


def _scores():
    return pd.DataFrame(
        {
            QUERY_ID: ["q1", "q1", "q1", "q2"],
            INCHIKEY14: [KEY_A, KEY_B, KEY_C, KEY_A],
            SCORE: [0.2, 0.9, 0.5, 1.0],
        }
    )


def test_valid_scores_pass():
    df = _scores()
    assert validate_candidate_scores(df) is df


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda d: d.drop(columns=[SCORE]), "missing columns"),
        (lambda d: d.assign(**{INCHIKEY14: ["short", KEY_B, KEY_C, KEY_A]}), "invalid inchikey14"),
        (lambda d: d.assign(**{SCORE: [None, 0.1, 0.2, 0.3]}), "nulls"),
        (lambda d: d.assign(**{SCORE: [float("inf"), 0.1, 0.2, 0.3]}), "finite"),
        (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate"),
    ],
)
def test_invalid_scores_rejected(mutate, msg):
    with pytest.raises(SchemaError, match=msg):
        validate_candidate_scores(mutate(_scores()))


def test_max_per_query():
    with pytest.raises(SchemaError, match="max_per_query"):
        validate_candidate_scores(_scores(), max_per_query=2)


def test_rank_candidates_orders_and_truncates():
    ranked = rank_candidates(_scores(), top_k=2)
    q1 = ranked[ranked[QUERY_ID] == "q1"]
    assert list(q1[INCHIKEY14]) == [KEY_B, KEY_C]
    assert list(q1[RANK]) == [1, 2]
    assert len(ranked) == 3
    assert TOP_K == 25


def test_round_trip(tmp_path):
    path = write_scores(_scores(), "demo.arch1", "val", root=tmp_path)
    assert path == tmp_path / "scores" / "demo.arch1" / "val.parquet"
    pd.testing.assert_frame_equal(read_scores("demo.arch1", "val", root=tmp_path), _scores())


@pytest.mark.parametrize("model_id, split", [("a/b", "val"), ("", "val"), ("ok", "holdout")])
def test_scores_path_rejects_bad_args(model_id, split, tmp_path):
    with pytest.raises(ValueError):
        scores_path(model_id, split, root=tmp_path)


def test_to_inchikey14():
    assert to_inchikey14("BSYNRYMUTXBXSQ-UHFFFAOYSA-N") == KEY_A
    with pytest.raises(SchemaError):
        to_inchikey14("not-a-key")


def test_spectrum_record_length_check():
    SpectrumRecord("s1", "q1", [100.0, 200.0], [1.0, 0.5], 250.0)
    with pytest.raises(SchemaError):
        SpectrumRecord("s1", "q1", [100.0], [1.0, 0.5], 250.0)


def test_model_runner_protocol():
    class Dummy:
        model_id = "dummy.arch1"

        def fit(self, train, val=None):
            pass

        def predict(self, spectra, candidates):
            return _scores()

    assert isinstance(Dummy(), ModelRunner)
