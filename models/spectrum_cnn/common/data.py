"""Notebook binning and Morgan targets, isolated from other members' TODO modules."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from casmi.preprocess.binning import bin_spectrum

RDKIT_VERSION = "2026.03.3"


def chemistry():
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    if rdBase.rdkitVersion != RDKIT_VERSION:
        raise RuntimeError(f"Expected RDKit {RDKIT_VERSION}; found {rdBase.rdkitVersion}. Install rdkit==2026.3.3.")
    return Chem, DataStructs, rdFingerprintGenerator


@lru_cache(maxsize=30000)
def fingerprint(smiles, fp_size=1024, radius=2):
    chem, structures, generators = chemistry()
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    generator = generators.GetMorganGenerator(radius=radius, fpSize=fp_size, includeChirality=False)
    vector = np.zeros(fp_size, dtype=np.uint8)
    structures.ConvertToNumpyArray(generator.GetFingerprint(mol), vector)
    return vector


def spectrum_to_bins(mzs, intensities, mz_max=2200, bin_width=1.0):
    return bin_spectrum(mzs, intensities, max_mz=mz_max, bin_width=bin_width, sqrt_intensity=True)


def read_spectra(path, *, targets=False):
    frame = pd.read_parquet(path)
    # Explicit aliases; query IDs are never inferred from row numbers.
    for original, canonical in [
        ("ms2_mzs", "mzs"),
        ("ms2_normalized_intensities", "intensities"),
        ("normalized_smiles", "smiles"),
    ]:
        if canonical not in frame and original in frame:
            frame = frame.rename(columns={original: canonical})
    required = {"mzs", "intensities"} | ({"smiles", "inchikey14"} if targets else set())
    if missing := required - set(frame.columns):
        raise ValueError(f"{path}: missing spectrum columns {sorted(missing)}")
    if frame.empty or frame[list(required)].isnull().any().any():
        raise ValueError(f"{path}: empty spectra or null required fields")
    return frame


def check_split(train, val, path):
    split = json.loads(path.read_text())
    train_ids, val_ids = set(split["train_structures"]), set(split["val_structures"])
    if train_ids & val_ids:
        raise ValueError("Structure split contains overlapping train/validation IDs")
    if not set(train.inchikey14) <= train_ids or not set(val.inchikey14) <= val_ids:
        raise ValueError("Dataset structures do not belong to the configured split")
    if set(train.inchikey14) & set(val.inchikey14):
        raise ValueError("Training and validation structures overlap")


class SpectrumDataset(Dataset):
    def __init__(self, frame, model_cfg, *, targets=False):
        if frame.attrs.get("casmi_preprocess", {}).get("sqrt_intensity"):
            raise ValueError("CNN expects raw intensities: input is already square-root transformed")
        self.records = frame.to_dict("records")
        self.cfg, self.targets = model_cfg, targets

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row, cfg = self.records[index], self.cfg
        vector = spectrum_to_bins(row["mzs"], row["intensities"], cfg["mz_max"], cfg["bin_width"])
        x = torch.from_numpy(vector).unsqueeze(0)
        if not self.targets:
            return x
        y = fingerprint(row["smiles"], cfg["fp_size"], cfg["fp_radius"]).astype(np.float32)
        return x, torch.from_numpy(y)
