"""MRR@25 scorer tests with hand-computed answers. Owner: Member 5.

Marked xfail until casmi.evaluation.scorer is implemented; remove the marker then.
"""

import pandas as pd
import pytest

from casmi.evaluation.scorer import mrr_at_k

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="scorer not implemented yet")

A, B, C = "BSYNRYMUTXBXSQ", "RZVAJINKPMORJF", "RYYVLZVUVIJVGH"


def test_mrr_hand_computed():
    scores = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2", "q3"],
            "inchikey14": [A, B, A, C, B],
            "score": [0.9, 0.1, 0.2, 0.8, 1.0],
        }
    )
    truth = pd.DataFrame({"query_id": ["q1", "q2", "q3"], "inchikey14": [A, A, C]})
    # q1: rank 1 -> 1.0, q2: rank 2 -> 0.5, q3: true key absent -> 0.0
    assert mrr_at_k(scores, truth) == pytest.approx(0.5)


def test_rank_beyond_k_scores_zero():
    keys = [f"{chr(65 + i // 26)}{chr(65 + i % 26)}{'X' * 12}" for i in range(30)]
    scores = pd.DataFrame({"query_id": "q1", "inchikey14": keys, "score": range(30, 0, -1)})
    truth = pd.DataFrame({"query_id": ["q1"], "inchikey14": [keys[26]]})  # rank 27
    assert mrr_at_k(scores, truth, k=25) == 0.0


def test_missing_query_counts_as_zero():
    scores = pd.DataFrame({"query_id": ["q1"], "inchikey14": [A], "score": [1.0]})
    truth = pd.DataFrame({"query_id": ["q1", "q2"], "inchikey14": [A, B]})
    assert mrr_at_k(scores, truth) == pytest.approx(0.5)
