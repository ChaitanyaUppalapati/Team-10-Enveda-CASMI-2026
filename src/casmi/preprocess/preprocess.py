"""Spectrum cleanup pipeline.

Owner: Member 2. Schedule task: "Preprocessing pipeline (label filter, peak cleanup,
deisotope, sqrt)". EDA findings (adducts, collision energy, precursor error, peak counts)
decide the defaults in ``PreprocessConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PreprocessConfig:
    min_rel_intensity: float = 0.01
    max_peaks: int = 150
    remove_above_precursor: bool = True
    deisotope: bool = True
    sqrt_intensity: bool = True


def filter_labels(train: pd.DataFrame) -> pd.DataFrame:
    """Drop training rows with missing or unparseable structure labels."""
    raise NotImplementedError("TODO(Member 2): filter invalid labels")


def clean_peaks(
    mzs: np.ndarray, intensities: np.ndarray, precursor_mz: float, cfg: PreprocessConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Remove noise peaks, peaks above precursor, keep top ``max_peaks``."""
    raise NotImplementedError("TODO(Member 2): peak cleanup")


def deisotope(mzs: np.ndarray, intensities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse isotope peaks onto their monoisotopic peak."""
    raise NotImplementedError("TODO(Member 2): deisotoping")


def preprocess_spectra(df: pd.DataFrame, cfg: PreprocessConfig | None = None) -> pd.DataFrame:
    """Apply the full pipeline to a spectrum table; returns the same schema."""
    raise NotImplementedError("TODO(Member 2): full preprocessing pipeline")
