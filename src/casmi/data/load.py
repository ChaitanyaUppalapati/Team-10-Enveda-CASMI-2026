"""Competition Parquet readers and explicit schema adaptation. Owner: Member 2.

Train has no native query IDs in the recorded raw schema: generated row IDs
are for per-spectrum training only, never frozen validation query identity.
Test molecule_id is the native query key. No query is inferred from labels.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from casmi.interfaces import SPECTRUM_COLUMNS, TRAIN_LABEL_COLUMNS, SchemaError
from casmi.utils.config import load_paths

from .schema import check_spectra

ALIASES = {
    "mzs": "ms2_mzs",
    "intensities": "ms2_normalized_intensities",
    "smiles": "normalized_smiles",
    "collision_energy": "collision_energy_ev",
}


def resolve_parquet(split: str, path: str | Path | None = None) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if path is not None:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        return target
    # load_paths currently overrides data_dir but not raw_dir; honor the data root here.
    if data_root := os.environ.get("CASMI_DATA_DIR"):
        root = Path(data_root).expanduser().resolve()
        candidates = [root / f"{split}.parquet", root / "raw" / f"{split}.parquet"]
    else:
        candidates = [load_paths()["raw_dir"] / f"{split}.parquet"]
    existing = [p for p in candidates if p.is_file()]
    if len(existing) > 1:
        raise ValueError(f"Multiple {split} files found; specify path explicitly: {existing}")
    if not existing:
        raise FileNotFoundError(f"No competition {split}.parquet found; checked {candidates}. No download was started.")
    return existing[0]


def _ids(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any() or series.map(lambda value: not str(value).strip()).any():
        raise SchemaError(f"{name} contains null/empty IDs")
    return series.map(str)


def normalize_spectra(
    raw: pd.DataFrame,
    *,
    split: str,
    row_offset: int = 0,
    query_map: pd.DataFrame | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Map raw names, preserve order/extras, and expose ID provenance in attrs.

    collision_energy_ev is list-valued in the recorded source and is retained
    as a list in collision_energy, not silently reduced to a scalar.
    """
    if split not in {"train", "test"} or row_offset < 0:
        raise ValueError("Expected train/test split and nonnegative row_offset")
    frame = raw.copy()
    for canonical, original in ALIASES.items():
        if canonical not in frame and original in frame:
            frame[canonical] = frame[original]
    if "spectrum_id" not in frame:
        if split == "test":
            raise SchemaError("Test spectra require native spectrum_id; no test IDs are generated")
        frame["spectrum_id"] = [f"train:row:{i}" for i in range(row_offset, row_offset + len(frame))]
        spectrum_source = "generated_physical_row_ordinal"
    else:
        spectrum_source = "native_spectrum_id"
    frame["spectrum_id"] = _ids(frame.spectrum_id, "spectrum_id")
    if frame.spectrum_id.duplicated().any():
        raise SchemaError("spectrum_id must be unique within a batch")
    if query_map is not None:
        if not {"spectrum_id", "query_id"} <= set(query_map):
            raise SchemaError("query_map requires spectrum_id and query_id")
        mapping = query_map[["spectrum_id", "query_id"]].copy()
        for name in mapping:
            mapping[name] = _ids(mapping[name], f"query_map.{name}")
        if mapping.spectrum_id.duplicated().any():
            raise SchemaError("query_map must have exactly one query per spectrum")
        frame["query_id"] = frame.spectrum_id.map(mapping.set_index("spectrum_id").query_id)
        query_source = "explicit_query_map"
    elif "query_id" in frame:
        query_source = "native_query_id"
    elif split == "test" and "molecule_id" in frame:
        frame["query_id"] = frame.molecule_id
        query_source = "native_molecule_id"
    elif split == "train":
        frame["query_id"] = frame.spectrum_id
        query_source = "per_spectrum_training_only"
    else:
        raise SchemaError("Test requires native query_id/molecule_id or an explicit query_map")
    frame["query_id"] = _ids(frame.query_id, "query_id")
    for optional in ("adduct", "collision_energy"):
        if optional not in frame:
            frame[optional] = None
    # Dropping labels is deliberate for test, even when a local fixture has them.
    if split == "test":
        frame = frame.drop(columns=["smiles", "normalized_smiles", "inchikey14", "inchikey"], errors="ignore")
    requested = list(columns) if columns is not None else None
    if requested is not None:
        if len(set(requested)) != len(requested):
            raise ValueError("Duplicate requested columns")
        if missing := set(requested) - set(frame):
            raise SchemaError(f"Missing requested columns: {sorted(missing)}")
        frame = frame[requested]
    if set(SPECTRUM_COLUMNS) <= set(frame):
        check_spectra(frame)
    elif requested is None:
        raise SchemaError(f"Missing spectrum columns: {sorted(set(SPECTRUM_COLUMNS) - set(frame))}")
    if split == "train" and requested is None and (missing := set(TRAIN_LABEL_COLUMNS) - set(frame)):
        raise SchemaError(f"Training table missing label columns: {sorted(missing)}")
    frame.attrs.update(
        {
            "split": split,
            "spectrum_id_source": spectrum_source,
            "query_id_source": query_source,
            "official_query_mapping": query_source != "per_spectrum_training_only",
        }
    )
    return frame


def iter_spectra(
    split: str,
    columns: list[str] | None = None,
    *,
    path: str | Path | None = None,
    batch_size: int = 10000,
    query_map: pd.DataFrame | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream raw rows in physical order, with stable IDs across batch sizes."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target = resolve_parquet(split, path)
    parquet = pq.ParquetFile(target)
    names = set(parquet.schema_arrow.names)
    selected = None
    if columns is not None:
        sources = set()
        for column in columns:
            if column in names:
                sources.add(column)
            elif ALIASES.get(column) in names:
                sources.add(ALIASES[column])
            elif column not in {"spectrum_id", "query_id", "adduct", "collision_energy"}:
                raise SchemaError(f"Raw Parquet has no source for requested column {column!r}")
        sources.update(names & {"spectrum_id", "query_id", "molecule_id"})
        # Keep a lightweight column so generated-ID-only projections retain row counts.
        selected = sorted(sources) or [parquet.schema_arrow.names[0]]
    offset = 0
    seen_ids = set()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
        raw = batch.to_pandas()
        # Check native spectrum ID uniqueness across batches, not only within one.
        if "spectrum_id" in raw:
            ids = set(_ids(raw.spectrum_id, "spectrum_id"))
            if seen_ids & ids:
                raise SchemaError("Duplicate native spectrum_id across batches")
            seen_ids.update(ids)
        frame = normalize_spectra(raw, split=split, row_offset=offset, query_map=query_map, columns=columns)
        frame.attrs["source_path"] = str(target)
        yield frame
        offset += len(raw)


def _load(split, columns, path, query_map):
    batches = list(iter_spectra(split, columns, path=path, query_map=query_map))
    if not batches:
        raise SchemaError("Competition Parquet contains no rows")
    result = pd.concat(batches, ignore_index=True)
    result.attrs = batches[0].attrs.copy()
    return result


def load_train(columns: list[str] | None = None, *, path=None, query_map=None) -> pd.DataFrame:
    """Load training spectra/labels; use iter_spectra for the full large dataset."""
    return _load("train", columns, path, query_map)


def load_test(columns: list[str] | None = None, *, path=None, query_map=None) -> pd.DataFrame:
    """Load unlabeled test spectra, grouping native molecule_id into query_id."""
    return _load("test", columns, path, query_map)
