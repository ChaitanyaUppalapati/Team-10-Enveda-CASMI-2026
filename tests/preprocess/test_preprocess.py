"""Preprocessing binning regression tests. Owner: Member 2."""

import numpy as np
import pytest

from casmi.preprocess.binning import bin_spectrum


def test_bin_spectrum_shape_and_max_pool():
    vec = bin_spectrum(np.array([100.01, 100.04, 500.0]), np.array([0.2, 0.7, 1.0]), max_mz=1000.0, bin_width=0.1)
    assert vec.shape == (10000,)
    assert vec[1000] == pytest.approx(0.7)
