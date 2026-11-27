"""Numpy-only features: spectrum binning, precursor/adduct/CE features, formula counts. Owner: Member 4."""

from __future__ import annotations

import re

import numpy as np

from casmi.candidates.lookup import ADDUCT_DELTAS, neutral_mass

MAX_MZ = 1200.0
BIN_WIDTH = 1.0
N_BINS = int(np.ceil(MAX_MZ / BIN_WIDTH))


def n_bins(bin_width: float = BIN_WIDTH) -> int:
    return int(np.ceil(MAX_MZ / bin_width))


def input_dim(bin_width: float = BIN_WIDTH, losses: bool = False) -> int:
    """Spectrum vector length: m/z bins, plus the same number of neutral-loss bins when ``losses``."""
    return n_bins(bin_width) * (2 if losses else 1)


def spec_args(data_cfg: dict) -> dict:
    """Spectrum-representation settings from the ``data`` config section."""
    return {"bin_width": data_cfg.get("bin_width", BIN_WIDTH), "losses": bool(data_cfg.get("neutral_loss", False))}

PEAK_MARGIN_DA = 2.0
"""Peaks above precursor + 2 Da were removed from the test set; do the same for training."""

ADDUCTS = tuple(ADDUCT_DELTAS)
ELEMENTS = ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I")
N_META = 2 + (len(ADDUCTS) + 1) + 1 + 2
CE_REF = np.log1p(100.0)


# ---------- spectra ----------
def bin_flat_peaks(
    mzs: np.ndarray,
    intensities: np.ndarray,
    offsets: np.ndarray,
    precursor_mz: np.ndarray,
    max_mz: float = MAX_MZ,
    bin_width: float = BIN_WIDTH,
    losses: bool = False,
) -> np.ndarray:
    """Vectorised sqrt + max-pool binning for many spectra stored as flat arrays plus offsets.

    Same result as ``casmi.preprocess.binning.bin_spectrum(..., sqrt_intensity=True)`` per spectrum,
    after dropping peaks above ``precursor_mz + PEAK_MARGIN_DA``. Returns float32 ``(n, n_bins)``; with
    ``losses`` a second block of the same width bins the neutral loss ``precursor_mz - m/z`` (peaks at or
    below the precursor), giving ``(n, 2 * n_bins)``.
    """
    n, n_bins = len(offsets) - 1, int(np.ceil(max_mz / bin_width))
    counts = np.diff(offsets)
    row = np.repeat(np.arange(n), counts)
    mzs, intensities = np.asarray(mzs, dtype=np.float64), np.asarray(intensities, dtype=np.float32)
    valid = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
        & (mzs >= 0)
        & (mzs < max_mz)
        & (intensities >= 0)
        & (mzs <= np.asarray(precursor_mz, dtype=np.float64)[row] + PEAK_MARGIN_DA)
    )
    out = np.zeros(n * n_bins, dtype=np.float32)
    flat = row[valid] * n_bins + np.floor(mzs[valid] / bin_width).astype(np.int64)
    np.maximum.at(out, flat, np.sqrt(intensities[valid]))
    out = out.reshape(n, n_bins)
    if not losses:
        return out
    loss = np.asarray(precursor_mz, dtype=np.float64)[row] - mzs
    ok = valid & (loss >= 0) & (loss < max_mz)
    out_loss = np.zeros(n * n_bins, dtype=np.float32)
    flat = row[ok] * n_bins + np.floor(loss[ok] / bin_width).astype(np.int64)
    np.maximum.at(out_loss, flat, np.sqrt(intensities[ok]))
    return np.concatenate([out, out_loss.reshape(n, n_bins)], axis=1)


# ---------- precursor / adduct / collision energy ----------
def mean_collision_energy(value) -> float:
    """Collision energy in eV: a scalar, a list of scalars (merged acquisitions -> mean), or missing."""
    if value is None:
        return float("nan")
    arr = np.asarray(value, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def meta_features(precursor_mz, adducts, collision_energy) -> np.ndarray:
    """``(n, N_META)`` features: m/z, neutral mass, adduct one-hot (+other), polarity, CE, CE-missing."""
    mz = np.asarray(precursor_mz, dtype=np.float64)
    n = len(mz)
    neutral = np.array([neutral_mass(m, a) for m, a in zip(mz, adducts, strict=True)])
    neutral = np.where(np.isfinite(neutral), neutral, mz)
    onehot = np.zeros((n, len(ADDUCTS) + 1), dtype=np.float32)
    for i, a in enumerate(adducts):
        onehot[i, ADDUCTS.index(a) if a in ADDUCTS else len(ADDUCTS)] = 1.0
    positive = np.array([1.0 if isinstance(a, str) and a.rstrip().endswith("+") else 0.0 for a in adducts])
    ce = np.asarray(collision_energy, dtype=np.float64)
    missing = ~np.isfinite(ce)
    ce_scaled = np.where(missing, 0.0, np.log1p(np.clip(np.nan_to_num(ce), 0, None)) / CE_REF)
    cols = [mz / MAX_MZ, neutral / MAX_MZ]
    return np.column_stack([*cols, onehot, positive, ce_scaled, missing.astype(np.float64)]).astype(np.float32)


# ---------- formula ----------
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_counts(formula: str) -> np.ndarray:
    """Counts of ``ELEMENTS`` in a molecular formula such as ``C8H10N4O2`` (others ignored)."""
    counts = dict.fromkeys(ELEMENTS, 0)
    for element, n in _FORMULA_TOKEN.findall(formula or ""):
        if element in counts:
            counts[element] += int(n) if n else 1
    return np.array([counts[e] for e in ELEMENTS], dtype=np.float32)


def log_counts(counts: np.ndarray) -> np.ndarray:
    """Auxiliary regression target / comparison space: log(1 + count)."""
    return np.log1p(counts)


def soft_tanimoto(consensus: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Tanimoto between a probability-like vector ``(n_bits,)`` and binary candidates ``(m, n_bits)``."""
    c = bits.astype(np.float32)
    dot = c @ consensus
    return dot / (float(consensus @ consensus) + c.sum(axis=1) - dot + 1e-9)
