"""Candidate retrieval by precursor mass / adduct. Owner: Member 4."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from casmi.interfaces import QUERY_ID

PROTON_MASS = 1.007276
ELECTRON_MASS = 0.00054858

# Monoisotopic atomic masses (Da).
_H, _NH3, _H2O = 1.00782503, 17.02654910, 18.01056468
_NA, _K, _CL = 22.98976928, 38.96370649, 34.96885268
_FORMIC_ACID = 46.00547930

# Adduct -> mass added to the neutral M to reach the observed ion m/z (charge 1).
# Positive ions lose an electron, negative ions gain one.
ADDUCT_DELTAS: dict[str, float] = {
    "[M+H]+": _H - ELECTRON_MASS,
    "[M+NH4]+": _NH3 + _H - ELECTRON_MASS,
    "[M-H2O+H]+": _H - _H2O - ELECTRON_MASS,
    "[M-2H2O+H]+": _H - 2 * _H2O - ELECTRON_MASS,
    "[M+Na]+": _NA - ELECTRON_MASS,
    "[M+K]+": _K - ELECTRON_MASS,
    "[M-H]-": -_H + ELECTRON_MASS,
    "[M-H2O-H]-": -_H2O - _H + ELECTRON_MASS,
    "[M+CH2O2-H]-": _FORMIC_ACID - _H + ELECTRON_MASS,
    "[M+Cl]-": _CL + ELECTRON_MASS,
}
"""The 10 adducts present in the hidden test set (see docs/competition_summary.md)."""

logger = logging.getLogger(__name__)


def neutral_mass(precursor_mz: float, adduct: str | None) -> float:
    """Neutral monoisotopic mass implied by a precursor m/z and adduct; ``nan`` if unknown."""
    delta = ADDUCT_DELTAS.get(adduct) if adduct else None
    return float("nan") if delta is None else float(precursor_mz) - delta


def candidates_for_queries(spectra: pd.DataFrame, pool: pd.DataFrame, ppm: float = 10.0) -> pd.DataFrame:
    """Return the pool rows whose mass matches each query's precursor.

    ``spectra`` needs ``query_id``, ``precursor_mz`` and ``adduct``; ``pool`` needs the standard pool
    columns. A query with several spectra gets the union of their matches, keeping the smallest
    ``ppm_error`` per candidate. Spectra with an unknown adduct are skipped and counted in a warning.

    Returns ``query_id, inchikey14, smiles, formula, monoisotopic_mass, ppm_error``.
    """
    if ppm <= 0:
        raise ValueError("ppm must be positive")
    out_cols = [QUERY_ID, *pool.columns, "ppm_error"]
    masses = np.array([neutral_mass(mz, ad) for mz, ad in zip(spectra["precursor_mz"], spectra["adduct"], strict=True)])
    known = np.isfinite(masses)
    if (n_skip := int((~known).sum())):
        logger.warning("skipping %d spectra with an unknown adduct", n_skip)

    order = np.argsort(pool["monoisotopic_mass"].to_numpy(), kind="stable")
    sorted_pool = pool.iloc[order].reset_index(drop=True)
    pool_mass = sorted_pool["monoisotopic_mass"].to_numpy()

    window = masses[known] * ppm * 1e-6
    lo = np.searchsorted(pool_mass, masses[known] - window, side="left")
    hi = np.searchsorted(pool_mass, masses[known] + window, side="right")
    counts = hi - lo
    if counts.sum() == 0:
        return pd.DataFrame(columns=out_cols)

    spec_pos = np.repeat(np.arange(known.sum()), counts)
    pool_pos = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi, strict=True) if b > a])
    hits = sorted_pool.iloc[pool_pos].reset_index(drop=True)
    hits.insert(0, QUERY_ID, spectra[QUERY_ID].to_numpy()[known][spec_pos])
    target = masses[known][spec_pos]
    hits["ppm_error"] = np.abs(hits["monoisotopic_mass"].to_numpy() - target) / target * 1e6
    hits = hits.sort_values([QUERY_ID, "inchikey14", "ppm_error"]).drop_duplicates([QUERY_ID, "inchikey14"])
    return hits.reset_index(drop=True)[out_cols]
