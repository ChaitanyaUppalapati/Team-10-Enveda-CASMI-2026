"""Fragment-formula explanation: can a candidate's molecular formula produce the observed fragment peaks?

Every fragment is made of atoms the molecule has. For a candidate formula we enumerate its heavy-atom
sub-formulas (C, N, O, P, S, halogens), allow any hydrogen count up to the parent's, and check whether
each strong peak matches one of them as a protonated / deprotonated / radical ion (or keeps the Na / K
of a metal adduct). The score is the share of peak intensity explained, in [0, 1].

Candidates that share a formula share the score, so a window of ~8,000 PubChem compounds needs only
the few dozen distinct formulas in it. Wrong-formula candidates (same mass, different atoms) score low;
true isomers of the answer score the same as the answer and are left to the fingerprint model.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

H = 1.00782503207
PROTON = 1.00727646688
ELECTRON = 0.00054857990907
NA, K = 22.98976928, 38.96370649
HEAVY = {"C": 12.0, "N": 14.0030740048, "O": 15.99491461956, "P": 30.97376163, "S": 31.972071,
         "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473}
MAX_SUBFORMULAS = 400_000  # guard for very large formulas


def parse_formula(formula):
    """'C10H15ClN2' -> {'C': 10, 'H': 15, 'Cl': 1, 'N': 2}; None for charged or unsupported formulas."""
    f = str(formula)
    if not f or any(c in f for c in "+-.") or not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", f):
        return None
    counts = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        if el != "H" and el not in HEAVY:
            return None
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts


@lru_cache(maxsize=200_000)
def _heavy_masses(formula):
    """Sorted masses of every heavy-atom sub-formula, and the parent's hydrogen count."""
    counts = parse_formula(formula)
    if counts is None:
        return None, 0
    masses = np.zeros(1)
    for el, m in HEAVY.items():
        n = counts.get(el, 0)
        if n:
            masses = (masses[:, None] + np.arange(n + 1) * m).ravel()
            if len(masses) > MAX_SUBFORMULAS:
                return None, 0
    return np.sort(masses), counts.get("H", 0)


def ion_shifts(adduct, mode):
    """Neutral-fragment mass = m/z - shift. Protonated or radical ions, plus the adduct's metal if any."""
    a = str(adduct)
    if str(mode).startswith("neg") or a.endswith("-"):
        return [-PROTON, ELECTRON]
    shifts = [PROTON, -ELECTRON]
    if "Na" in a:
        shifts.append(NA - ELECTRON)
    if "K" in a and "NH4" not in a:
        shifts.append(K - ELECTRON)
    return shifts


def molecule_peaks(rows, mz, intensity, adducts, modes, top=20, floor=0.02):
    """Strong peaks of one molecule's spectra: (neutral-fragment masses per shift, weights).

    rows: positions of the molecule's spectra in mz / intensity. Each spectrum's top peaks get equal total
    weight, so a molecule with many spectra is not dominated by one of them.
    """
    targets, weights = [], []
    for r in rows:
        inten = intensity[r]
        n = int((inten > 0).sum())
        if n == 0:
            continue
        x, w = mz[r, :n], inten[:n]
        keep = np.argsort(-w)[:top]
        keep = keep[w[keep] >= floor * w.max()]
        x, w = x[keep], w[keep] / w[keep].sum()
        shifts = np.array(ion_shifts(adducts[r], modes[r]))
        targets.append(x[:, None] - shifts[None, :])
        weights.append(w)
    if not targets:
        return None
    width = max(t.shape[1] for t in targets)  # pad shift columns with NaN so spectra stack
    targets = [np.pad(t, ((0, 0), (0, width - t.shape[1])), constant_values=np.nan) for t in targets]
    w = np.concatenate(weights)
    return np.concatenate(targets), w / w.sum()


def explained_fraction(formula, peaks, ppm=10.0, abs_tol=0.002):
    """Share of peak weight explained by sub-formulas of `formula` (nan if the formula is unusable)."""
    hm, hmax = _heavy_masses(formula)
    if hm is None or peaks is None:
        return np.nan
    targets, weights = peaks
    ok = np.zeros(len(targets), dtype=bool)
    for j in range(targets.shape[1]):
        t = targets[:, j]
        valid = np.isfinite(t)
        if not valid.any():
            continue
        tol = np.maximum(abs_tol, t[valid] * ppm * 1e-6)
        # for each target, every heavy sub-formula lighter than it; the rest must be hydrogens
        hi = np.searchsorted(hm, t[valid] + tol, side="right")
        for k, (tv, tl, h_end) in enumerate(zip(t[valid], tol, hi, strict=True)):
            res = tv - hm[:h_end]
            nh = np.rint(res / H)
            hit = (nh >= 0) & (nh <= hmax + 1) & (np.abs(res - nh * H) <= tl)
            if hit.any():
                ok[np.flatnonzero(valid)[k]] = True
    return float(weights[ok].sum())


def window_scores(formulas, peaks, cache=None, **kw):
    """Explanation score for each candidate in a window (candidates with the same formula share it)."""
    cache = {} if cache is None else cache
    out = np.empty(len(formulas))
    for i, f in enumerate(formulas):
        if f not in cache:
            cache[f] = explained_fraction(f, peaks, **kw)
        out[i] = cache[f]
    return out


def has_elements(formula, elements=("F", "I")):
    counts = parse_formula(formula)
    return bool(counts) and any(counts.get(e, 0) for e in elements)
