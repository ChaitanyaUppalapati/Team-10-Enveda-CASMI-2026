"""Adducts, neutral masses and Morgan fingerprints (RDKit imported lazily)."""

from __future__ import annotations

import numpy as np

# Singly charged adduct mass shifts (neutral M -> observed m/z), keyed without brackets.
ADDUCT_DELTA = {
    "M+H+": 1.007276, "M+Na+": 22.989218, "M+K+": 38.963158, "M+NH4+": 18.033823,
    "M-H2O+H+": -17.00274, "M+H-H2O+": -17.00274, "M-H-": -1.007276, "M+Cl-": 34.969402,
    "M+FA-H-": 44.998201, "M+HCOO-": 44.998201, "M+Na-2H-": 20.974666,
}


def adduct_delta(adduct) -> float:
    """Mass shift for ``adduct``; unknown/missing falls back to [M+H]+ (or [M-H]- if it ends in '-')."""
    text = str(adduct or "").replace(" ", "").replace("[", "").replace("]", "")
    if text in ADDUCT_DELTA:
        return ADDUCT_DELTA[text]
    return ADDUCT_DELTA["M-H-"] if text.endswith("-") else ADDUCT_DELTA["M+H+"]


def neutral_mass(precursor_mz: float, adduct) -> float:
    return float(precursor_mz) - adduct_delta(adduct)


def exact_mass(smiles: str) -> float:
    """Monoisotopic mass of a SMILES string, NaN if RDKit cannot parse it."""
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt

    mol = Chem.MolFromSmiles(smiles)
    return float(ExactMolWt(mol)) if mol is not None else float("nan")


def morgan_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan count-free bit vector as uint8; all zeros if RDKit cannot parse the SMILES."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    out = np.zeros(n_bits, dtype=np.uint8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), out)
    return out
