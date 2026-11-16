"""Small IO helpers. Owner: Member 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet file, optionally only some columns."""
    return pd.read_parquet(path, columns=columns)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to parquet, creating parent directories. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
