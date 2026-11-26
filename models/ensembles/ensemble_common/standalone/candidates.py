"""External candidate structures (COCONUT, PubChemLite) and pool fingerprints, computed in parallel.

Any attached file whose path names a source (``coconut`` or ``pubchem``, which includes PubChemLite) is
read. Its SMILES column is parsed with RDKit (largest fragment kept), and each structure becomes
(inchikey14, smiles, exact mass, packed 2048-bit Morgan r2). The frame plugs into Member 3's
``build_pool(..., extra_candidates=...)``. Same idea and the same Kaggle datasets as Member 7's v2
notebook, so pools match.
"""

from __future__ import annotations

import glob
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

SOURCES = ("coconut", "pubchem")  # "pubchem" also matches PubChemLite files
SMILES_COLUMNS = ("canonical_smiles", "smiles", "isomeric_smiles", "normalized_smiles")
EXTENSIONS = (".csv", ".csv.gz", ".tsv", ".zip", ".parquet")
MASS_RANGE = (100.0, 1500.0)  # hidden test molecules span about 157-1,159 Da


def find_candidate_files(root: str = "/kaggle/input") -> dict:
    """{source: [files]} for attached files whose path names a source."""
    found: dict[str, list[str]] = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        low = f.lower()
        if os.path.isfile(f) and low.endswith(EXTENSIONS):
            for name in SOURCES:
                if name in low:
                    found.setdefault(name, []).append(f)
    return found


def _smiles_column(columns) -> str:
    low = {str(c).lower(): c for c in columns}
    for c in SMILES_COLUMNS:
        if c in low:
            return low[c]
    for c in columns:
        if "smiles" in str(c).lower():
            return c
    raise ValueError(f"no SMILES column in {list(columns)[:20]}")


def read_smiles(path: str) -> np.ndarray:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
        return df[_smiles_column(df.columns)].dropna().astype(str).unique()
    sep = "\t" if ".tsv" in path.lower() else ","
    header = pd.read_csv(path, sep=sep, nrows=0)
    if len(header.columns) == 1 and "\t" in str(header.columns[0]):
        sep = "\t"
        header = pd.read_csv(path, sep=sep, nrows=0)
    col = _smiles_column(header.columns)
    df = pd.read_csv(path, sep=sep, usecols=[col], dtype=str, on_bad_lines="skip")
    return df[col].dropna().astype(str).unique()


def _convert(args):
    """Worker: SMILES chunk -> (keys, canonical smiles, masses, packed fingerprints) for valid structures."""
    chunk, n_bits = args
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    chooser = rdMolStandardize.LargestFragmentChooser()
    keys, smis, masses, fps = [], [], [], []
    for s in chunk:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            if len(Chem.GetMolFrags(mol)) > 1:
                mol = chooser.choose(mol)
            mass = Descriptors.ExactMolWt(mol)
            if not MASS_RANGE[0] <= mass <= MASS_RANGE[1]:
                continue
            key = Chem.MolToInchiKey(mol)[:14]
            if len(key) != 14:
                continue
            keys.append(key)
            smis.append(Chem.MolToSmiles(mol))
            masses.append(mass)
            fps.append(np.packbits(gen.GetFingerprintAsNumPy(mol).astype(bool)))
        except Exception:  # noqa: BLE001 - one bad record must not stop the run
            continue
    fp = np.stack(fps) if fps else np.zeros((0, n_bits // 8), dtype=np.uint8)
    return keys, smis, masses, fp


def _fp_only(args):
    """Worker: SMILES chunk -> packed fingerprints in the same order (zeros when RDKit cannot parse)."""
    chunk, n_bits = args
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    out = np.zeros((len(chunk), n_bits // 8), dtype=np.uint8)
    for i, s in enumerate(chunk):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                out[i] = np.packbits(gen.GetFingerprintAsNumPy(mol).astype(bool))
        except Exception:  # noqa: BLE001
            continue
    return out


def _map(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with get_context("fork").Pool(workers) as p:
        return p.map(fn, jobs)


def load_external(files: dict, n_bits: int = 2048, workers: int | None = None, chunk: int = 20_000, log=print):
    """files: {source: [paths]}. Returns (frame[key, smiles, mass, source], packed fp) with unique keys."""
    workers = workers or os.cpu_count() or 1
    frames, fps = [], []
    for source, paths in files.items():
        t0 = time.time()
        smiles = np.unique(np.concatenate([read_smiles(p) for p in paths]))
        parts = _map(_convert, [(smiles[i : i + chunk].tolist(), n_bits) for i in range(0, len(smiles), chunk)],
                     workers)
        frame = pd.DataFrame({"key": [k for p in parts for k in p[0]], "smiles": [s for p in parts for s in p[1]],
                              "mass": [m for p in parts for m in p[2]]})
        frame["source"] = source
        frames.append(frame)
        fps.append(np.concatenate([p[3] for p in parts]))
        log(f"{source}: {len(smiles):,} SMILES -> {len(frame):,} structures in range in {time.time() - t0:.0f}s")
    if not frames:
        return None, None
    frame = pd.concat(frames, ignore_index=True)
    fp = np.concatenate(fps)
    keep = ~frame.key.duplicated().to_numpy()
    return frame[keep].reset_index(drop=True), fp[keep]


def pool_fingerprints(pool: pd.DataFrame, extra: pd.DataFrame | None, extra_fp: np.ndarray | None,
                      n_bits: int = 2048, workers: int | None = None, chunk: int = 10_000) -> np.ndarray:
    """Packed Morgan fingerprints for every pool row: external rows reuse ``extra_fp``, the rest use RDKit."""
    workers = workers or os.cpu_count() or 1
    fp = np.zeros((len(pool), n_bits // 8), dtype=np.uint8)
    todo = np.ones(len(pool), dtype=bool)
    if extra is not None and len(extra):
        is_ext = (pool.formula.astype(str) == "").to_numpy()
        where = pd.Series(np.arange(len(extra)), index=extra.key.to_numpy())
        hit = pool.key.map(where).fillna(-1).to_numpy().astype(np.int64)
        hit[~is_ext] = -1
        fp[hit >= 0] = extra_fp[hit[hit >= 0]]
        todo &= hit < 0
    rows = np.flatnonzero(todo)
    smiles = pool.smiles.to_numpy()[rows].tolist()
    parts = _map(_fp_only, [(smiles[i : i + chunk], n_bits) for i in range(0, len(smiles), chunk)], workers)
    if parts:
        fp[rows] = np.concatenate(parts)
    return fp
