"""Structure standardisation shared by every candidate pool. Owner: Member 4.

One function turns a raw SMILES into the pool schema, so COCONUT, PubChem and any future source
agree on the match key. The steps mirror ``models/spectra_pooling/common/candidates.py`` (Member 7):
largest fragment, RDKit InChIKey, first 14 characters.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

POOL_COLUMNS = ("inchikey14", "smiles", "formula", "monoisotopic_mass")
MASS_RANGE = (100.0, 1500.0)
"""Neutral monoisotopic mass window kept in a pool (the hidden test spans 157-1,159 Da)."""

logger = logging.getLogger(__name__)


def standardize_smiles(smiles: str, mass_range: tuple[float, float] = MASS_RANGE) -> tuple | None:
    """Return ``(inchikey14, smiles, formula, monoisotopic_mass)`` or ``None`` if unusable.

    ``None`` means the SMILES did not parse, produced no InChIKey, or falls outside ``mass_range``.
    Multi-component records keep only their largest fragment (salts, solvents).
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if len(Chem.GetMolFrags(mol)) > 1:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mass = Descriptors.ExactMolWt(mol)
        if not mass_range[0] <= mass <= mass_range[1]:
            return None
        key = Chem.MolToInchiKey(mol)[:14]
        if len(key) != 14:
            return None
        return key, Chem.MolToSmiles(mol), rdMolDescriptors.CalcMolFormula(mol), mass
    except Exception:  # noqa: BLE001 - one bad record must not stop a million-row build
        return None


def _standardize_chunk(args: tuple[list[str], tuple[float, float]]) -> list[tuple]:
    chunk, mass_range = args
    return [rec for smi in chunk if isinstance(smi, str) and (rec := standardize_smiles(smi, mass_range))]


def standardize_many(
    smiles: Iterable[str],
    mass_range: tuple[float, float] = MASS_RANGE,
    workers: int | None = 1,
    chunk_size: int = 20_000,
) -> pd.DataFrame:
    """Standardise many SMILES into a pool table with unique ``inchikey14`` (first record wins).

    ``workers`` > 1 spreads chunks over processes (``0`` or ``None`` = all cores). Output order and
    content do not depend on ``workers``.
    """
    items = list(smiles)
    workers = (os.cpu_count() or 1) if not workers else workers
    chunks = [(items[i : i + chunk_size], mass_range) for i in range(0, len(items), chunk_size)]
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_standardize_chunk, chunks))
    else:
        parts = [_standardize_chunk(c) for c in chunks]
    rows = [rec for part in parts for rec in part]
    pool_df = pd.DataFrame(rows, columns=list(POOL_COLUMNS))
    pool_df = pool_df.drop_duplicates("inchikey14", keep="first").reset_index(drop=True)
    logger.info("standardised %d SMILES -> %d unique structures", len(items), len(pool_df))
    return pool_df
