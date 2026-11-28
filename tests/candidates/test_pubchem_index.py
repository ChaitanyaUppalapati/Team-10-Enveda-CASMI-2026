"""Tests for the mass-sorted PubChem index reader and builder. Owner: Member 4."""

import gzip

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmi.candidates.pubchem_index import COLUMNS, MassIndex, build_index


def _write_index(path, masses, row_group_size=3):
    df = pd.DataFrame(
        {
            "cid": np.arange(len(masses)),
            "mass": np.sort(np.asarray(masses, dtype=float)),
            "formula": ["C1"] * len(masses),
            "pubmed": np.arange(len(masses), dtype=np.int32),
            "smiles": [f"C{i}" for i in range(len(masses))],
        }
    )
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=row_group_size)
    return df


def test_window_returns_exactly_the_rows_within_ppm_across_row_groups(tmp_path):
    masses = [100.0, 150.0, 200.0, 200.001, 200.002, 200.003, 250.0, 300.0, 400.0]
    df = _write_index(tmp_path / "idx.parquet", masses, row_group_size=2)
    idx = MassIndex(tmp_path / "idx.parquet")
    got = idx.window(200.001, ppm=10.0)  # +-0.002 Da: 200.000 ... 200.003 all qualify
    assert got["mass"].tolist() == [200.0, 200.001, 200.002, 200.003]
    assert list(got.columns) == list(COLUMNS)
    tight = idx.window(200.001, ppm=0.5)  # +-0.0001 Da
    assert tight["mass"].tolist() == [200.001]
    assert idx.window(123.0).empty and idx.num_rows == len(df)


def test_window_sizes_counts_without_reading_other_columns(tmp_path):
    _write_index(tmp_path / "idx.parquet", [100.0, 200.0, 200.0001, 300.0])
    idx = MassIndex(tmp_path / "idx.parquet")
    assert idx.window_sizes([200.0, 100.0, 999.0], ppm=10.0).tolist() == [2, 1, 0]


def test_build_index_joins_filters_charged_species_and_sorts(tmp_path):
    pytest.importorskip("duckdb")
    raw = tmp_path / "raw"
    raw.mkdir()
    mass = ["1\tC9H17NO4\t203.1157\t203.1157", "2\tC9H18NO4+\t204.1236\t204.1236", "3\tC7H8O4\t156.0423\t156.0423",
            "4\tCH4\t16.0313\t16.0313", "5\tC10H12\t132.0939\t132.0939"]  # fmt: skip
    smiles = ["1\tCC(C)(C)OC(=O)NCC(=O)O", "2\tC[NH+](C)C", "3\tOC1C=CC(=O)C=C1O", "4\tC", "5\tC1CCc2ccccc2C1"]
    pmid = ["1\t111\t1", "1\t222\t1", "3\t333\t1"]
    for name, rows in (("CID-Mass.gz", mass), ("CID-SMILES.gz", smiles), ("CID-PMID.gz", pmid)):
        with gzip.open(raw / name, "wt") as fh:
            fh.write("\n".join(rows) + "\n")
    out = build_index(
        raw, tmp_path / "idx.parquet", mass_range=(100.0, 1500.0), row_group_size=2, memory_limit="1GB", threads=1
    )
    df = pq.read_table(out).to_pandas()
    assert df["cid"].tolist() == [5, 3, 1]  # CH4 (too light) and the cation are gone; sorted by mass
    assert df.set_index("cid")["pubmed"].to_dict() == {5: 0, 3: 1, 1: 2}
