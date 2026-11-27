"""External candidate structures (COCONUT, PubChemLite) and the merged candidate pool.

Each external SMILES is parsed with RDKit, reduced to its largest fragment (salts and counter-ions
would otherwise change the metric key), tautomer-canonicalised to the official key, and given its exact
mass. Sources are merged on the official key, keeping per-source evidence as features:

- ``n_spectra``: training spectra of the structure (0 if not in training, or if held out)
- ``in_coconut``, ``np_likeness``, ``coconut_annot``: COCONUT membership and natural-product score
- ``in_pcl``, ``pubmed``, ``patent``, ``annot_types``: PubChemLite membership and literature counts
- ``in_pc``, ``pc_pmid``: membership of the PubChem literature slice and its PubMed reference count
"""

from __future__ import annotations

from multiprocessing import Pool as _MP

import numpy as np
import pandas as pd

from .data import INTERIM, REPO

EXTERNAL = REPO / "data" / "external"
SOURCES = {  # file, SMILES column, mass column (for the pre-filter), extra columns kept as features
    "coconut": (EXTERNAL / "coconut" / "coconut_csv_lite-11-2026.csv", "canonical_smiles", "exact_molecular_weight",
                {"np_likeness": "np_likeness", "annotation_level": "coconut_annot"}),
    "pcl": (EXTERNAL / "pubchemlite" / "PubChemLite_exposomics_20260925.csv", "SMILES", "MonoisotopicMass",
            {"PubMed_Count": "pubmed", "Patent_Count": "patent", "AnnoTypeCount": "annot_types"}),
    # PubChem compounds with literature (built by scripts/make_pubchem_slice.py from PubChem Extras)
    "pc": (EXTERNAL / "pubchem" / "pubchem_slice.csv", "SMILES", "mass", {"pmid_count": "pc_pmid"}),
}
# Hidden test masses are 157-1159 Da. The pre-filter (on the file's own mass, which may include salts) is
# looser; the exact range is applied after RDKit. Very large molecules are also the slow ones to canonicalise.
MASS_RANGE = (100.0, 1250.0)
PREFILTER_MAX = 1400.0
_ENUM = None


def _process(smiles: str):
    """SMILES -> (largest-fragment SMILES, official key, exact mass), or None."""
    global _ENUM
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    if _ENUM is None:
        _ENUM = rdMolStandardize.TautomerEnumerator()
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        try:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        except Exception:
            pass
    try:
        key_mol = _ENUM.Canonicalize(mol)
    except Exception:
        key_mol = mol
    key = Chem.MolToInchiKey(key_mol)
    if not key:
        return None
    return Chem.MolToSmiles(mol), key[:14], Descriptors.ExactMolWt(mol)


def load_external(name: str, refresh: bool = False, n_jobs: int = 9) -> pd.DataFrame:
    path = INTERIM / f"cand_{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    import time

    file, smi_col, mass_col, extra = SOURCES[name]
    raw = pd.read_csv(file, usecols=[smi_col, mass_col, *extra])
    raw = raw[(raw[mass_col] >= MASS_RANGE[0]) & (raw[mass_col] <= PREFILTER_MAX)]
    # largest first, so the slow molecules do not all land at the end of the run
    raw = raw.assign(_len=raw[smi_col].str.len()).sort_values("_len", ascending=False).reset_index(drop=True)
    smiles = raw[smi_col].tolist()
    res, t0 = [], time.time()
    with _MP(n_jobs) as mp:
        for i, r in enumerate(mp.imap(_process, smiles, chunksize=64)):
            res.append(r)
            if (i + 1) % 50000 == 0:
                print(f"  {name}: {i + 1:,}/{len(smiles):,} ({time.time() - t0:.0f}s)", flush=True)
    ok = np.array([r is not None for r in res])
    out = pd.DataFrame([r for r in res if r is not None], columns=["smiles", "key", "mass"])
    for src, dst in extra.items():
        out[dst] = raw.loc[ok, src].to_numpy()
    out = out[(out.mass >= MASS_RANGE[0]) & (out.mass <= MASS_RANGE[1])]
    sort_col = next(iter(extra.values()))  # keep the best-evidenced row per key
    out = out.sort_values(sort_col, ascending=False).drop_duplicates("key").reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


def build_pool_table(structures: pd.DataFrame, exclude_keys=(), sources=("coconut", "pcl")) -> pd.DataFrame:
    """Merged candidate table, one row per official key. ``exclude_keys`` are inchikey14s held out of training."""
    tr = structures[~structures.inchikey14.isin(set(exclude_keys))]
    tr = (tr.sort_values("n_spectra", ascending=False).groupby("key", as_index=False)
          .agg(smiles=("smiles", "first"), mass=("mass", "first"), n_spectra=("n_spectra", "sum")))
    parts = [tr.assign(in_train=1)]
    for s in sources:
        e = load_external(s).assign(**{f"in_{s}": 1})
        parts.append(e)
    t = pd.concat(parts, ignore_index=True)
    agg = {"smiles": "first", "mass": "first"}
    num = [c for c in t.columns if c not in ("key", "smiles", "mass")]
    agg.update({c: "max" for c in num})
    t = t.groupby("key", as_index=False, sort=False).agg(agg)  # first = training SMILES when present
    for c in num:
        t[c] = t[c].fillna(0)
    t["prior"] = default_prior(t)
    return t


def default_prior(t: pd.DataFrame) -> np.ndarray:
    """Tie-break among same-mass isomers before any learned ranker: measured, then well-known structures."""
    pubmed = t["pubmed"] if "pubmed" in t else 0
    patent = t["patent"] if "patent" in t else 0
    return (np.log1p(t["n_spectra"]) + 0.5 * np.log1p(pubmed + patent)).to_numpy()
