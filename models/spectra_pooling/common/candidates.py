"""External candidates (COCONUT, PubChemLite) and gated B2 fusion for the offline Kaggle notebook.

Candidates: every attached file whose path contains a source name is read, its SMILES column is parsed
with RDKit (largest fragment kept), and each structure becomes (inchikey14, smiles, exact mass, Morgan
fingerprint). Parsing runs in worker processes; ~1M structures take a few minutes on Kaggle's 4 CPUs.

Gate: B2 (library search) is excellent when the structure has library spectra and useless otherwise.
Use the B2 ranking only for molecules whose best library match scores at least tau, else the model's.
tau is chosen on validation; the honest estimate is the 2-fold cross-fitted score.
"""

from __future__ import annotations

import glob
import os
import time
import zlib
from multiprocessing import get_context

import numpy as np
import pandas as pd

SOURCES = ("coconut", "pubchemlite")
SMILES_COLUMNS = ("canonical_smiles", "smiles", "isomeric_smiles", "normalized_smiles")
EXTENSIONS = (".csv", ".csv.gz", ".zip", ".parquet", ".tsv")
MASS_RANGE = (100.0, 1500.0)  # hidden test spans 157-1,159 Da


# ---------- external candidates ----------


def find_files(root="/kaggle/input"):
    """{source: [files]} for attached files whose path names a source."""
    found = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        low = f.lower()
        if os.path.isfile(f) and low.endswith(EXTENSIONS):
            for name in SOURCES:
                if name in low:
                    found.setdefault(name, []).append(f)
    return found


def smiles_column(columns):
    low = {c.lower(): c for c in columns}
    for c in SMILES_COLUMNS:
        if c in low:
            return low[c]
    for c in columns:
        if "smiles" in c.lower():
            return c
    raise ValueError(f"no SMILES column in {list(columns)[:20]}")


def read_smiles(path):
    """Unique SMILES strings from a csv / tsv / zip / parquet file."""
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
        return df[smiles_column(df.columns)].dropna().astype(str).unique()
    sep = "\t" if ".tsv" in path.lower() else ","
    header = pd.read_csv(path, sep=sep, nrows=0)
    if len(header.columns) == 1 and "\t" in header.columns[0]:
        sep = "\t"
        header = pd.read_csv(path, sep=sep, nrows=0)
    col = smiles_column(header.columns)
    df = pd.read_csv(path, sep=sep, usecols=[col], dtype=str, on_bad_lines="skip")
    return df[col].dropna().astype(str).unique()


def _convert(args):
    """Worker: SMILES chunk -> keys, canonical smiles, masses, packed fingerprints."""
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


def load_candidates(files, n_bits=2048, workers=None, chunk=20000, log=print):
    """files: {source: [paths]}. Returns (frame[key, smiles, mass, source], packed fp), unique keys."""
    frames, fps = [], []
    workers = workers or os.cpu_count() or 1
    for source, paths in files.items():
        t0 = time.time()
        smiles = np.unique(np.concatenate([read_smiles(p) for p in paths]))
        jobs = [(smiles[i:i + chunk].tolist(), n_bits) for i in range(0, len(smiles), chunk)]
        with get_context("fork").Pool(workers) as pool:
            parts = pool.map(_convert, jobs)
        keys = [k for p in parts for k in p[0]]
        frame = pd.DataFrame({"key": keys, "smiles": [s for p in parts for s in p[1]],
                              "mass": [m for p in parts for m in p[2]], "source": source})
        frames.append(frame)
        fps.append(np.concatenate([p[3] for p in parts]))
        log(f"{source}: {len(smiles):,} SMILES -> {len(frame):,} structures in range "
            f"({frame.key.nunique():,} unique) in {time.time() - t0:.0f}s")
    if not frames:
        return None, None
    frame = pd.concat(frames, ignore_index=True)
    fp = np.concatenate(fps)
    keep = ~frame.key.duplicated().to_numpy()
    return frame[keep].reset_index(drop=True), fp[keep]


# ---------- gated B2 fusion ----------


def b2_top_scores(cp, pool, library, spectra, spectra_mz, spectra_in, b2_rank, modified=True):
    """Per molecule: mean over its spectra of the best library match for B2's top candidate.

    Same number B2 ranked by (pooling="mean"), recomputed for one candidate only. 0 = no library match.
    """
    keys = pool.key.to_numpy()
    spectra = spectra.assign(pos=np.arange(len(spectra)))
    out = {}
    for mol, group in spectra.groupby("molecule_id", sort=False):
        top = keys[b2_rank[mol][0]]
        values = []
        for row in group.itertuples():
            n = int((spectra_in[row.pos] > 0).sum())
            best = 0.0
            for lib_pos in library.index.get((top, row.ionization_mode), ()) if n else ():
                m = library.n_peaks[lib_pos]
                if m == 0:
                    continue
                shift = (row.precursor_mz - library.precursor[lib_pos]) if modified else None
                best = max(best, cp.match_score(spectra_mz[row.pos, :n], spectra_in[row.pos, :n],
                                                library.mz[lib_pos, :m], library.intensity[lib_pos, :m], shift=shift))
            values.append(best)
        out[mol] = float(np.mean(values)) if values else 0.0
    return out


def gate(model_rank, alt_rank, scores, tau):
    """alt ranking where the library match is strong (score >= tau), the model's elsewhere."""
    return {m: (alt_rank[m] if scores.get(m, 0.0) >= tau else r) for m, r in model_rank.items()}


def _balanced(cp, rankings, truth, pool_keys):
    return float(cp.evaluate(rankings, truth, pool_keys).loc["balanced", "MRR@25"])


def tune_gate(cp, truth, pool_keys, model_rank, alts, scores, taus=None):
    """Pick (alt, tau) on validation by balanced MRR@25.

    alts: {name: rankings} to switch to (e.g. B2, fusion). Returns a dict with the chosen alt and tau,
    the in-sample score, and a 2-fold cross-fitted score (tau tuned on one half, applied to the other).
    """
    s = np.array([scores.get(m, 0.0) for m in truth.molecule_id])
    taus = taus if taus is not None else np.unique(np.r_[np.quantile(s[s > 0], np.linspace(0, 1, 41)) if
                                                          (s > 0).any() else [], np.inf])
    fold = np.array([zlib.crc32(m.encode()) % 2 for m in truth.molecule_id])

    def best_on(mask):
        sub = truth[mask]
        best = (-1.0, None, np.inf)
        for name, alt in alts.items():
            for tau in taus:
                v = _balanced(cp, gate(model_rank, alt, scores, tau), sub, pool_keys)
                if v > best[0]:
                    best = (v, name, tau)
        return best

    score_in, name, tau = best_on(np.ones(len(truth), dtype=bool))
    crossed = {}
    for f in (0, 1):
        _, n_f, t_f = best_on(fold == f)
        held = truth.molecule_id[fold != f]
        g = gate({m: model_rank[m] for m in held}, alts[n_f], scores, t_f)
        crossed.update(g)
    return {"alt": name, "tau": float(tau), "in_sample": round(score_in, 4),
            "cross_fitted": round(_balanced(cp, crossed, truth, pool_keys), 4),
            "share_gated_val": round(float((s >= tau).mean()), 4)}
