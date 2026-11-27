"""Data access: cached metadata, per-structure table, and spectra read by row id."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .chem import formula_mass, neutral_mass, official_keys, ppm

REPO = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("CASMI_DATA_DIR", REPO / "data" / "raw"))
INTERIM = Path(os.environ.get("CASMI_INTERIM_DIR", REPO / "data" / "interim"))
OUTPUTS = Path(os.environ.get("CASMI_OUTPUTS_DIR", REPO / "outputs"))
TRAIN = RAW / "train.parquet"
TEST = RAW / "test.parquet"

META_COLS = ["ingest_lib", "inchikey14", "molecular_formula", "ionization_mode", "instrument_type", "adduct",
             "precursor_mz", "precursor_error_ppm", "num_peaks", "collision_energy_ev"]
MAX_LABEL_PPM = 20.0  # spectra whose precursor disagrees more than this with the label are treated as bad


def load_meta(refresh: bool = False) -> pd.DataFrame:
    """One row per training spectrum (no peaks), with ``row_id`` = row position in train.parquet."""
    path = INTERIM / "meta.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    INTERIM.mkdir(parents=True, exist_ok=True)
    m = pq.read_table(TRAIN, columns=META_COLS).to_pandas()
    m.insert(0, "row_id", np.arange(len(m), dtype=np.int64))
    m["ce_mean"] = m.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
    m = m.drop(columns="collision_energy_ev")
    m["neutral_mass"] = neutral_mass(m.precursor_mz.to_numpy(), m.adduct.to_numpy())
    m["good_label"] = m.precursor_error_ppm.abs().fillna(np.inf) <= MAX_LABEL_PPM
    for c in ("ingest_lib", "ionization_mode", "instrument_type", "adduct"):
        m[c] = m[c].astype("category")
    m.to_parquet(path, index=False)
    return m


def load_structures(meta: pd.DataFrame | None = None, refresh: bool = False, n_jobs: int = 8) -> pd.DataFrame:
    """One row per training structure: SMILES, formula, exact mass, official metric key, spectrum counts."""
    path = INTERIM / "structures.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    meta = load_meta() if meta is None else meta
    smi = pq.read_table(TRAIN, columns=["inchikey14", "normalized_smiles"]).to_pandas()
    smi = smi.drop_duplicates("inchikey14").set_index("inchikey14").normalized_smiles
    g = meta.groupby("inchikey14", observed=True)
    s = pd.DataFrame({
        "formula": g.molecular_formula.first(),
        "n_spectra": g.size(),
        "n_libs": g.ingest_lib.nunique(),
    })
    s["smiles"] = smi.reindex(s.index).to_numpy()
    s["mass"] = s.formula.map(formula_mass)
    s["key"] = official_keys(s.smiles.tolist(), n_jobs=n_jobs)
    s = s.reset_index().rename(columns={"inchikey14": "inchikey14"})
    s.to_parquet(path, index=False)
    return s


class SpectrumReader:
    """Read peak lists for arbitrary row ids of a parquet file, one row group at a time."""

    def __init__(self, path: Path = TRAIN):
        self.file = pq.ParquetFile(path)
        sizes = [self.file.metadata.row_group(i).num_rows for i in range(self.file.num_row_groups)]
        self.starts = np.concatenate([[0], np.cumsum(sizes)])

    def read(self, row_ids) -> tuple[list[np.ndarray], list[np.ndarray]]:
        row_ids = np.asarray(row_ids, dtype=np.int64)
        mzs: list = [None] * len(row_ids)
        ints: list = [None] * len(row_ids)
        group = np.searchsorted(self.starts, row_ids, side="right") - 1
        for gi in np.unique(group):
            sel = np.flatnonzero(group == gi)
            t = self.file.read_row_group(int(gi), columns=["ms2_mzs", "ms2_normalized_intensities"])
            local = row_ids[sel] - self.starts[gi]
            mz_col = t.column(0).take(local).to_pylist()
            in_col = t.column(1).take(local).to_pylist()
            for j, a, b in zip(sel, mz_col, in_col, strict=True):
                mzs[j] = np.asarray(a, dtype=np.float32)
                ints[j] = np.asarray(b, dtype=np.float32)
        return mzs, ints

    def iter_groups(self, columns):
        """Yield (row_offset, table) per row group: for full passes over the file."""
        for gi in range(self.file.num_row_groups):
            yield int(self.starts[gi]), self.file.read_row_group(gi, columns=columns)


def load_test() -> pd.DataFrame:
    t = pd.read_parquet(TEST)
    t["neutral_mass"] = neutral_mass(t.precursor_mz.to_numpy(), t.adduct.to_numpy())
    return t


__all__ = ["RAW", "INTERIM", "OUTPUTS", "TRAIN", "TEST", "load_meta", "load_structures", "SpectrumReader",
           "load_test", "ppm"]
