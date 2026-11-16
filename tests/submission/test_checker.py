"""Submission checker tests. Owner: Member 6.

Marked xfail until casmi.submission.checker is implemented; remove the marker then.
"""

import pandas as pd
import pytest

from casmi.interfaces import SchemaError
from casmi.submission.checker import check_submission

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="checker not implemented yet")


def test_empty_submission_rejected():
    with pytest.raises(SchemaError):
        check_submission(pd.DataFrame(), test_query_ids={"q1"})
