"""Validate submission.csv before uploading. Owner: Member 6.

Schedule task: "Submission checker + offline Kaggle notebook skeleton (B0)".
"""

from __future__ import annotations

import pandas as pd


def check_submission(sub: pd.DataFrame, test_query_ids: set[str] | None = None) -> pd.DataFrame:
    """Raise SchemaError on format problems (columns, every test query present, <=25 each)."""
    raise NotImplementedError("TODO(Member 6): submission format checks")
