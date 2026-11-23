"""Explicit peak cleanup profiles; defaults preserve CNN inputs. Owner: Member 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from casmi.interfaces import INCHIKEY14_RE, SchemaError


@dataclass(frozen=True)
class PreprocessConfig:
    # Scaffold filters were never implemented; conservative defaults avoid changing trained inputs.
    min_rel_intensity: float = 0.0
    max_peaks: int | None = None
    remove_above_precursor: bool = False
    deisotope: bool = False
    sqrt_intensity: bool = False
    isotope_charge: int = 1
    isotope_tolerance_da: float = 0.005
    isotope_min_peaks: int = 3

    def __post_init__(self):
        if not np.isfinite(self.min_rel_intensity) or not 0 <= self.min_rel_intensity <= 1:
            raise ValueError("min_rel_intensity must be in [0, 1]")
        if self.max_peaks is not None and (not isinstance(self.max_peaks, int) or self.max_peaks < 1):
            raise ValueError("max_peaks must be a positive integer or None")
        if self.isotope_charge < 1 or not isinstance(self.isotope_charge, int):
            raise ValueError("isotope_charge must be a positive integer")
        if not np.isfinite(self.isotope_tolerance_da) or not 0 < self.isotope_tolerance_da < 0.1:
            raise ValueError("isotope_tolerance_da must be in (0, 0.1)")
        if not isinstance(self.isotope_min_peaks, int) or self.isotope_min_peaks < 3:
            raise ValueError("isotope_min_peaks must be at least 3")

    @classmethod
    def cnn_compatible(cls):
        """Keep valid raw intensities; the CNN performs square root inside its binning step."""
        return cls()


def filter_labels(train: pd.DataFrame) -> pd.DataFrame:
    """Drop missing/unparseable SMILES and invalid 14-letter label keys; preserve row order.

    This does not canonicalize labels or assert SMILES/InChIKey agreement.
    Those are separate scoring-key audit responsibilities.
    """
    from rdkit import Chem

    name = "smiles" if "smiles" in train else "normalized_smiles"
    if name not in train or "inchikey14" not in train:
        raise SchemaError("Label filtering requires smiles/normalized_smiles and inchikey14")
    valid_keys = train.inchikey14.map(lambda key: isinstance(key, str) and bool(INCHIKEY14_RE.fullmatch(key)))
    parsed = {}
    for smiles in train.loc[valid_keys, name].dropna().unique():
        parsed[smiles] = isinstance(smiles, str) and bool(smiles.strip()) and Chem.MolFromSmiles(smiles) is not None
    keep = valid_keys & train[name].map(lambda smiles: parsed.get(smiles, False))
    result = train.loc[keep].copy()
    result.attrs["label_filter"] = {
        "input_rows": len(train),
        "kept_rows": len(result),
        "dropped_rows": int((~keep).sum()),
    }
    return result


def _valid_peaks(mzs, intensities):
    mzs, intensities = np.asarray(mzs, dtype=np.float64), np.asarray(intensities)
    if mzs.ndim != 1 or intensities.ndim != 1 or mzs.shape != intensities.shape:
        raise ValueError("Peak arrays must be one-dimensional and equally sized")
    # Preserve floating precision for the identity profile; integer intensities become float64.
    if intensities.dtype.kind not in "f":
        intensities = intensities.astype(np.float64)
    valid = np.isfinite(mzs) & np.isfinite(intensities) & (mzs >= 0) & (intensities >= 0)
    return mzs[valid].copy(), intensities[valid].copy()


def deisotope(
    mzs: np.ndarray,
    intensities: np.ndarray,
    *,
    charge: int = 1,
    tolerance_da: float = 0.005,
    min_peaks: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Optional simple C13-spacing heuristic with explicit fixed fragment charge.

    Requires >=3 consecutive non-increasing positive peaks, collapses their
    summed intensity to the first m/z, and keeps unmatched peaks. Does not
    infer charge, convert to single charge, or implement the OpenMS algorithm.
    Disabled by default; chemical validity on CASMI has not been benchmarked.
    """
    PreprocessConfig(isotope_charge=charge, isotope_tolerance_da=tolerance_da, isotope_min_peaks=min_peaks)
    mzs, intensities = _valid_peaks(mzs, intensities)
    order = np.argsort(mzs, kind="stable")
    mzs, intensities = mzs[order], intensities[order]
    consumed = np.zeros(len(mzs), dtype=bool)
    out_mz, out_intensity = [], []
    spacing = 1.00335483507 / charge
    for index in range(len(mzs)):
        if consumed[index]:
            continue
        chain = [index]
        if intensities[index] > 0:
            for step in range(1, len(mzs)):
                target = mzs[index] + step * spacing
                left, right = np.searchsorted(mzs, [target - tolerance_da, target + tolerance_da], side="left")
                possible = [
                    j
                    for j in range(max(index + 1, left), right)
                    if not consumed[j] and j not in chain and 0 < intensities[j] <= intensities[chain[-1]]
                ]
                if not possible:
                    break
                chain.append(min(possible, key=lambda j: (abs(mzs[j] - target), j)))
        if len(chain) >= min_peaks:
            consumed[chain] = True
            out_mz.append(mzs[index])
            out_intensity.append(float(intensities[chain].sum()))
        else:
            consumed[index] = True
            out_mz.append(mzs[index])
            out_intensity.append(float(intensities[index]))
    return np.asarray(out_mz, dtype=np.float64), np.asarray(out_intensity, dtype=intensities.dtype)


def clean_peaks(
    mzs: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean peaks without mutation; sqrt is deliberately performed last and only if requested."""
    mzs, intensities = _valid_peaks(mzs, intensities)
    if cfg.remove_above_precursor:
        if not np.isfinite(precursor_mz) or precursor_mz <= 0:
            raise ValueError("A finite positive precursor_mz is required for precursor trimming")
        keep = mzs <= precursor_mz
        mzs, intensities = mzs[keep], intensities[keep]
    if len(intensities) and cfg.min_rel_intensity > 0:
        keep = intensities >= intensities.max() * cfg.min_rel_intensity
        mzs, intensities = mzs[keep], intensities[keep]
    if cfg.deisotope:
        mzs, intensities = deisotope(
            mzs,
            intensities,
            charge=cfg.isotope_charge,
            tolerance_da=cfg.isotope_tolerance_da,
            min_peaks=cfg.isotope_min_peaks,
        )
    if cfg.max_peaks is not None and len(mzs) > cfg.max_peaks:
        # Stable ties and original order among retained peaks.
        selected = np.sort(np.argsort(-intensities, kind="stable")[: cfg.max_peaks])
        mzs, intensities = mzs[selected], intensities[selected]
    if cfg.sqrt_intensity:
        intensities = np.sqrt(intensities)
    return mzs, intensities


def preprocess_spectra(df: pd.DataFrame, cfg: PreprocessConfig | None = None) -> pd.DataFrame:
    """Preserve IDs, labels, row order and schema; never filter labeled queries implicitly."""
    cfg = cfg or PreprocessConfig.cnn_compatible()
    if not {"mzs", "intensities", "precursor_mz"} <= set(df):
        raise SchemaError("Preprocessing requires mzs, intensities and precursor_mz")
    previous = df.attrs.get("casmi_preprocess", {})
    if previous.get("sqrt_intensity") and cfg.sqrt_intensity:
        raise ValueError("Input already square-root transformed; refusing a second square root")
    result = df.copy()
    peaks = [
        clean_peaks(mz, intensity, precursor, cfg)
        for mz, intensity, precursor in zip(df.mzs, df.intensities, df.precursor_mz, strict=True)
    ]
    result["mzs"] = [p[0] for p in peaks]
    result["intensities"] = [p[1] for p in peaks]
    # Retained raw-name aliases must not contradict the canonical transformed columns.
    if "ms2_mzs" in result:
        result["ms2_mzs"] = result["mzs"]
    if "ms2_normalized_intensities" in result:
        result["ms2_normalized_intensities"] = result["intensities"]
    result.attrs["casmi_preprocess"] = {
        **asdict(cfg),
        "sqrt_intensity": cfg.sqrt_intensity or previous.get("sqrt_intensity", False),
    }
    return result
