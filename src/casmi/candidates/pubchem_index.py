"""Mass-sorted PubChem index: build it from the bulk files, then read narrow mass windows. Owner: Member 4.

A candidate pool of ~100M PubChem structures is far too large to embed, so candidates are fetched per query: the
index is one parquet file sorted by mass with small row groups (min/max mass in the footer), so a +-10 ppm window
touches one or two row groups and a query reads a few hundred KB. Columns: ``cid, mass, formula, pubmed, smiles``.

Build (needs ``duckdb``; reads PubChem's ``CID-Mass.gz``, ``CID-SMILES.gz``, ``CID-PMID.gz`` from
https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/)::

    from casmi.candidates.pubchem_index import build_index
    build_index("data/candidates/raw_pubchem", "data/candidates/pubchem_index.parquet")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = ("cid", "mass", "formula", "pubmed", "smiles")


def build_index(
    raw_dir: str | Path,
    out_path: str | Path,
    mass_range: tuple[float, float] = (100.0, 1500.0),
    row_group_size: int = 10_000,
    memory_limit: str = "14GB",
    threads: int = 8,
    temp_dir: str | Path | None = None,
) -> Path:
    """Join mass, SMILES and PubMed counts, keep neutral compounds in ``mass_range``, sort by mass, write parquet.

    Charged species (formulas with ``+`` or ``-``) are dropped because queries assume a neutral molecule.
    ``pubmed`` is the number of PubMed articles linked to the compound (0 if none).
    """
    import duckdb

    raw, out = Path(raw_dir), Path(out_path)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'; SET threads={threads};")
    if temp_dir:
        con.execute(f"SET temp_directory='{temp_dir}';")
    lo, hi = mass_range
    con.execute(
        f"""
        COPY (
          SELECT m.cid, m.mass, m.formula, COALESCE(p.pubmed, 0)::INTEGER AS pubmed, s.smiles
          FROM read_csv('{raw / "CID-Mass.gz"}', delim='\\t', header=false, quote='',
                        columns={{'cid':'BIGINT','formula':'VARCHAR','mass':'DOUBLE','mono':'DOUBLE'}}) m
          JOIN read_csv('{raw / "CID-SMILES.gz"}', delim='\\t', header=false, quote='',
                        columns={{'cid':'BIGINT','smiles':'VARCHAR'}}) s USING (cid)
          LEFT JOIN (
            SELECT cid, count(*) AS pubmed
            FROM read_csv('{raw / "CID-PMID.gz"}', delim='\\t', header=false, quote='',
                          columns={{'cid':'BIGINT','pmid':'BIGINT','kind':'INTEGER'}})
            GROUP BY cid
          ) p USING (cid)
          WHERE m.mass BETWEEN {lo} AND {hi} AND m.formula NOT LIKE '%+%' AND m.formula NOT LIKE '%-%'
          ORDER BY m.mass, m.cid
        ) TO '{out}' (FORMAT PARQUET, ROW_GROUP_SIZE {row_group_size}, COMPRESSION ZSTD)
        """
    )
    return out


class MassIndex:
    """Read mass windows from an index written by :func:`build_index` (any parquet with COLUMNS, sorted by mass)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._file = pq.ParquetFile(self.path)
        meta = self._file.metadata
        names = [meta.schema.column(i).name for i in range(meta.num_columns)]
        j = names.index("mass")
        stats = [meta.row_group(g).column(j).statistics for g in range(meta.num_row_groups)]
        self.rg_min = np.array([s.min for s in stats], dtype=np.float64)
        self.rg_max = np.array([s.max for s in stats], dtype=np.float64)
        self.num_rows = meta.num_rows

    def window(self, mass: float, ppm: float = 10.0, columns: tuple[str, ...] = COLUMNS) -> pd.DataFrame:
        """Rows whose mass is within ``ppm`` of ``mass``, sorted by mass."""
        lo, hi = mass * (1 - ppm * 1e-6), mass * (1 + ppm * 1e-6)
        groups = np.flatnonzero((self.rg_max >= lo) & (self.rg_min <= hi))
        if len(groups) == 0:
            return pd.DataFrame({c: [] for c in columns})
        need = tuple(dict.fromkeys((*columns, "mass")))
        table = pa.concat_tables([self._file.read_row_group(int(g), columns=list(need)) for g in groups])
        df = table.to_pandas()
        df = df[(df["mass"] >= lo) & (df["mass"] <= hi)]
        return df[list(columns)].reset_index(drop=True)

    def window_sizes(self, masses, ppm: float = 10.0) -> np.ndarray:
        """Number of structures in each window (reads only the mass column)."""
        out = np.zeros(len(masses), dtype=np.int64)
        for i, mass in enumerate(masses):
            out[i] = len(self.window(float(mass), ppm, columns=("mass",)))
        return out
