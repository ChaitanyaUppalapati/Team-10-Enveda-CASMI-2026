"""Keep shared preprocessing metadata effective at the CNN file boundary."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from casmi.preprocess import PreprocessConfig, preprocess_spectra

pytest.importorskip("torch", reason="CNN boundary tests require the optional ml dependencies")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import SpectrumDataset, read_spectra  # noqa: E402


def test_sqrt_metadata_survives_parquet_and_cnn_rejects_it(tmp_path):
    frame = pd.DataFrame({"mzs": [[1.0, 2.0]], "intensities": [[4.0, 9.0]], "precursor_mz": [100.0]})
    cleaned = preprocess_spectra(frame, PreprocessConfig(sqrt_intensity=True))
    path = tmp_path / "transformed.parquet"
    cleaned.to_parquet(path, index=False)
    restored = read_spectra(path)
    with pytest.raises(ValueError, match="already square-root"):
        SpectrumDataset(restored, {})
