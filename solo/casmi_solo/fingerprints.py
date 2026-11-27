"""Morgan fingerprints (radius 2, 2048 bits), packed as uint8, computed in parallel and cached by key."""

from __future__ import annotations

from multiprocessing import Pool as _MP

import numpy as np
import pandas as pd

from .data import INTERIM

FP_BITS = 2048
FP_RADIUS = 2
_GEN = None


def _fp(smiles: str) -> bytes | None:
    global _GEN
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    if _GEN is None:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return np.packbits(_GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)).tobytes()


def packed_fingerprints(smiles: list[str], n_jobs: int = 9) -> np.ndarray:
    with _MP(n_jobs) as mp:
        res = mp.map(_fp, smiles, chunksize=1024)
    out = np.zeros((len(smiles), FP_BITS // 8), dtype=np.uint8)
    for i, r in enumerate(res):
        if r is not None:
            out[i] = np.frombuffer(r, dtype=np.uint8)
    return out


def fingerprint_store(table: pd.DataFrame, name: str, refresh: bool = False) -> tuple[np.ndarray, dict]:
    """Packed fingerprints for every key of ``table`` (columns key, smiles), cached as data/interim/fp_<name>.npz."""
    path = INTERIM / f"fp_{name}.npz"
    if path.exists() and not refresh:
        z = np.load(path, allow_pickle=True)
        keys, fp = z["keys"], z["fp"]
    else:
        keys = table.key.to_numpy().astype(object)
        fp = packed_fingerprints(table.smiles.tolist())
        np.savez(path, keys=keys, fp=fp)
    return fp, {k: i for i, k in enumerate(keys)}


def extended_store(base: tuple[np.ndarray, dict], table: pd.DataFrame, name: str) -> tuple[np.ndarray, dict]:
    """``base`` store plus fingerprints of the keys of ``table`` it lacks (those cached as fp_<name>_extra.npz)."""
    fp, index = base
    new = table[~table.key.isin(index.keys())].drop_duplicates("key")
    efp, eindex = fingerprint_store(new, f"{name}_extra")
    keys = list(index) + [k for k in eindex if k not in index]
    fp = np.concatenate([fp, efp[[eindex[k] for k in keys[len(index):]]]]) if len(keys) > len(index) else fp
    return fp, {k: i for i, k in enumerate(keys)}


def unpack(fp_packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(fp_packed, axis=-1)[..., :FP_BITS]
