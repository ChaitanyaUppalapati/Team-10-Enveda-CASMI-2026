"""Chemistry helpers: formula masses, adduct arithmetic and the metric's structure key."""

from __future__ import annotations

import re
from multiprocessing import Pool

import numpy as np

PROTON = 1.00727646688

# Monoisotopic masses of the elements that occur in the training formulas.
ELEMENT_MASS = {
    "H": 1.00782503207, "C": 12.0, "N": 14.0030740048, "O": 15.99491461956, "S": 31.97207100,
    "P": 30.97376163, "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473,
    "Si": 27.9769265325, "B": 11.0093054, "Se": 79.9165213, "Na": 22.9897692809, "K": 38.96370668,
    "As": 74.9215965, "Fe": 55.9349375, "Co": 58.933195, "Cu": 62.9295975, "Zn": 63.9291422,
    "Sn": 119.9021947, "Hg": 201.970643, "Pt": 194.9647911, "Mg": 23.9850417, "Ca": 39.96259098,
    "Li": 7.01600455, "Al": 26.98153863, "Ge": 73.9211778, "Te": 129.9062244, "Bi": 208.9803987,
    "Ti": 47.9479463, "Mn": 54.9380451, "Ni": 57.9353429, "Cr": 51.9405075, "Gd": 157.9241039,
    "Ag": 106.905097, "Au": 196.9665687, "Sb": 120.9038157, "Ru": 101.9043493, "Rh": 102.905504,
    "Pd": 105.903486, "W": 183.9509312, "Mo": 97.9054082, "V": 50.9439595, "Zr": 89.9047044,
    "Ba": 137.9052472, "Sr": 87.9056121, "Cs": 132.905451933, "Rb": 84.911789738, "Tl": 204.9744275,
    "Pb": 207.9766521, "Cd": 113.9033585, "In": 114.903878, "Ga": 68.9255736, "Be": 9.0121822,
    "D": 2.0141017778,
}
ELECTRON = 0.00054857990907
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

# adduct -> (molecules per ion, |charge|, mass added to n*M before dividing by |charge|)
H2O = 2 * ELEMENT_MASS["H"] + ELEMENT_MASS["O"]
ADDUCTS = {
    "[M+H]+": (1, 1, PROTON),
    "[M+NH4]+": (1, 1, ELEMENT_MASS["N"] + 4 * ELEMENT_MASS["H"] - ELECTRON),
    "[M-H2O+H]+": (1, 1, PROTON - H2O),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+Na]+": (1, 1, ELEMENT_MASS["Na"] - ELECTRON),
    "[M+K]+": (1, 1, ELEMENT_MASS["K"] - ELECTRON),
    "[M-H]-": (1, 1, -PROTON),
    "[M-H2O-H]-": (1, 1, -PROTON - H2O),
    "[M+CH2O2-H]-": (1, 1, ELEMENT_MASS["C"] + 2 * ELEMENT_MASS["H"] + 2 * ELEMENT_MASS["O"] - PROTON),
    "[M+Cl]-": (1, 1, ELEMENT_MASS["Cl"] + ELECTRON),
}
TEST_ADDUCTS = tuple(ADDUCTS)


def formula_mass(formula: str) -> float:
    """Monoisotopic neutral mass of a molecular formula such as ``C6H12O6``. NaN if unparseable."""
    if not isinstance(formula, str) or not formula:
        return float("nan")
    body = formula.split("+")[0].split("-")[0]  # drop a trailing charge if present
    total, pos = 0.0, 0
    for el, n in _TOKEN.findall(body):
        if el not in ELEMENT_MASS:
            return float("nan")
        total += ELEMENT_MASS[el] * (int(n) if n else 1)
        pos += len(el) + len(n)
    return total if pos == len(body) else float("nan")


def neutral_mass(precursor_mz, adduct):
    """Neutral monoisotopic mass from precursor m/z and adduct (vectorised; NaN for unknown adducts)."""
    mz = np.asarray(precursor_mz, dtype=np.float64)
    add = np.broadcast_to(np.asarray(adduct, dtype=object), mz.shape)
    out = np.full(mz.shape, np.nan)
    for name, (n, z, delta) in ADDUCTS.items():
        hit = add == name
        out[hit] = (mz[hit] * z - delta) / n
    return out


def ppm(observed, reference):
    return (np.asarray(observed) - np.asarray(reference)) / np.asarray(reference) * 1e6


# ---- The metric's structure key: RDKit tautomer canonicalisation, then the first InChIKey block ----
_ENUM = None


def official_key(smiles: str) -> str | None:
    """Key the CASMI metric compares: InChIKey14 of the RDKit tautomer-canonical molecule."""
    global _ENUM
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    if _ENUM is None:
        _ENUM = rdMolStandardize.TautomerEnumerator()
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        mol = _ENUM.Canonicalize(mol)
    except Exception:  # canonicalisation failure: fall back to the plain key
        pass
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


def official_keys(smiles: list[str], n_jobs: int = 8, chunksize: int = 256) -> list[str | None]:
    if n_jobs <= 1 or len(smiles) < 2000:
        return [official_key(s) for s in smiles]
    with Pool(n_jobs) as pool:
        return pool.map(official_key, smiles, chunksize=chunksize)
