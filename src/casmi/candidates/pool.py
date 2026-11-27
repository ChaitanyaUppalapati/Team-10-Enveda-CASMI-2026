"""Read raw structure exports and load built pool tables. Owner: Member 4."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from casmi.utils.config import load_paths

from .standardize import MASS_RANGE, POOL_COLUMNS, standardize_many

SMILES_COLUMNS = ("canonical_smiles", "smiles", "isomeric_smiles", "normalized_smiles")


def find_smiles_column(columns: list[str]) -> str:
    """Pick the SMILES column from a raw export, preferring the known names."""
    lower = {c.lower(): c for c in columns}
    for name in SMILES_COLUMNS:
        if name in lower:
            return lower[name]
    for c in columns:
        if "smiles" in c.lower():
            return c
    raise ValueError(f"no SMILES column in {list(columns)[:20]}")


def read_raw_smiles(path: str | Path) -> list[str]:
    """Unique SMILES strings from a csv / tsv / parquet export (``.gz`` allowed)."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        sep = "\t" if ".tsv" in name else ","
        df = pd.read_csv(path, sep=sep, dtype=str, on_bad_lines="skip")
    col = find_smiles_column(list(df.columns))
    return df[col].dropna().astype(str).unique().tolist()


def build_pool(
    source: str | Path, mass_range: tuple[float, float] = MASS_RANGE, workers: int | None = 0
) -> pd.DataFrame:
    """Standardise a raw export into the pool table (``POOL_COLUMNS``, unique ``inchikey14``).

    ``workers=0`` uses all cores.
    """
    return standardize_many(read_raw_smiles(source), mass_range, workers=workers)


def load_pool(filename: str, candidates_dir: str | Path | None = None) -> pd.DataFrame:
    """Load a built pool parquet from ``candidates_dir`` (defaults to the configured path)."""
    root = Path(candidates_dir) if candidates_dir is not None else load_paths()["candidates_dir"]
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; build it with kaggle/datasets/coconut/build_dataset.py")
    pool = pd.read_parquet(path)
    if missing := set(POOL_COLUMNS) - set(pool.columns):
        raise ValueError(f"{path} is missing pool columns: {sorted(missing)}")
    return pool[list(POOL_COLUMNS)]
