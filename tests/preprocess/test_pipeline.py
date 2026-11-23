"""Public preprocessing defaults, optional transforms and input parity. Owner: Member 2."""

import numpy as np
import pandas as pd
import pytest

from casmi.preprocess import (
    PreprocessConfig,
    bin_spectrum,
    clean_peaks,
    deisotope,
    filter_labels,
    preprocess_spectra,
)


def test_default_preserves_valid_peaks_and_uses_no_sqrt():
    mz = np.array([200, 10, 10.1], dtype=np.float64)
    intensity = np.array([0.81, 0.04, 0.64], dtype=np.float32)
    a, b = clean_peaks(mz, intensity, 100, PreprocessConfig.cnn_compatible())
    np.testing.assert_array_equal(a, mz)
    np.testing.assert_array_equal(b, intensity)
    assert not np.shares_memory(b, intensity)


def test_invalid_peaks_removed_and_empty_supported():
    mz, intensity = clean_peaks([1, np.nan, 3, -1, 5], [1, 1, -1, 1, np.inf], 100, PreprocessConfig())
    np.testing.assert_array_equal(mz, [1])
    np.testing.assert_array_equal(intensity, [1])
    assert bin_spectrum([], []).sum() == 0
    with pytest.raises(ValueError, match="equally sized"):
        clean_peaks([1], [], 100, PreprocessConfig())


def test_explicit_filters_and_sqrt():
    cfg = PreprocessConfig(min_rel_intensity=0.2, max_peaks=1, remove_above_precursor=True, sqrt_intensity=True)
    mz, intensity = clean_peaks([10, 20, 30, 101], [1, 25, 16, 100], 100, cfg)
    np.testing.assert_array_equal(mz, [20])
    np.testing.assert_array_equal(intensity, [5])


def test_pipeline_keeps_ids_labels_and_rejects_double_sqrt():
    frame = pd.DataFrame(
        {
            "query_id": ["q"],
            "spectrum_id": ["s"],
            "inchikey14": ["A" * 14],
            "mzs": [[1, 2]],
            "intensities": [[4.0, 9.0]],
            "precursor_mz": [100.0],
        }
    )
    first = preprocess_spectra(frame, PreprocessConfig(sqrt_intensity=True))
    assert first.query_id.tolist() == ["q"] and first.inchikey14.tolist() == ["A" * 14]
    assert frame.intensities.iloc[0] == [4.0, 9.0]
    with pytest.raises(ValueError, match="second square root"):
        preprocess_spectra(first, PreprocessConfig(sqrt_intensity=True))


def test_isotope_heuristic_requires_complete_envelope_and_preserves_unmatched():
    spacing = 1.00335483507
    mz, intensity = deisotope([100, 100 + spacing, 100 + 2 * spacing, 150], [10.0, 5.0, 2.0, 8.0])
    np.testing.assert_allclose(mz, [100, 150])
    np.testing.assert_allclose(intensity, [17, 8])
    mz, intensity = deisotope([100, 100 + spacing], [10.0, 5.0])
    assert len(mz) == 2  # A lone near-spacing pair is not enough.


def test_label_filter_does_not_canonicalize_or_modify_source():
    pytest.importorskip("rdkit", reason="Label parsing requires the optional chem dependencies")
    frame = pd.DataFrame(
        {"smiles": ["CCO", None, "not valid smiles", "CCN"], "inchikey14": ["A" * 14, "B" * 14, "C" * 14, "bad"]}
    )
    filtered = filter_labels(frame)
    assert filtered.index.tolist() == [0]
    assert filtered.smiles.tolist() == ["CCO"] and len(frame) == 4
    assert filtered.attrs["label_filter"]["dropped_rows"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_rel_intensity": -1},
        {"max_peaks": 0},
        {"isotope_charge": 0},
        {"isotope_tolerance_da": 0.2},
        {"isotope_min_peaks": 2},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        PreprocessConfig(**kwargs)
