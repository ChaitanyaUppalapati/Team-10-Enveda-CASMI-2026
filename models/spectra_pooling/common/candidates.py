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
MAX_SMILES_LEN = 400  # longer SMILES are polymers/salts/annotations, far outside the test chemistry


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
    """Worker: SMILES chunk -> keys, canonical smiles, masses, formulas, packed fingerprints."""
    chunk, n_bits = args
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    chooser = rdMolStandardize.LargestFragmentChooser()
    keys, smis, masses, formulas, fps, kept = [], [], [], [], [], []
    for pos, s in enumerate(chunk):
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
            formulas.append(CalcMolFormula(mol))
            fps.append(np.packbits(gen.GetFingerprintAsNumPy(mol).astype(bool)))
            kept.append(pos)
        except Exception:  # noqa: BLE001 - one bad record must not stop the run
            continue
    fp = np.stack(fps) if fps else np.zeros((0, n_bits // 8), dtype=np.uint8)
    return keys, smis, masses, formulas, fp, kept


def load_candidates(files, n_bits=2048, workers=None, chunk=20000, log=print, smiles_by_source=None):
    """files: {source: [paths]}; smiles_by_source: {source: SMILES array} already read (e.g. PubChem windows).

    Returns (frame[key, smiles, mass, source], packed fp) with unique keys; earlier sources win duplicates.
    """
    frames, fps = [], []
    workers = workers or os.cpu_count() or 1
    sources = {s: (lambda paths=paths: np.concatenate([read_smiles(p) for p in paths])) for s, paths in files.items()}
    for s, arr in (smiles_by_source or {}).items():
        sources[s] = (lambda arr=arr: np.asarray(arr, dtype=object))
    for source, get in sources.items():
        t0 = time.time()
        # pandas keeps Python strings; numpy .astype(str) would pad every SMILES to the longest one (OOM)
        smiles = pd.Series(get(), dtype=object).dropna().astype(str)
        smiles = smiles[smiles.str.len() <= MAX_SMILES_LEN].unique()
        jobs = [(smiles[i:i + chunk].tolist(), n_bits) for i in range(0, len(smiles), chunk)]
        with get_context("fork").Pool(workers) as pool:
            parts = pool.map(_convert, jobs)
        keys = [k for p in parts for k in p[0]]
        frame = pd.DataFrame({"key": keys, "smiles": [s for p in parts for s in p[1]],
                              "mass": [m for p in parts for m in p[2]], "formula": [f for p in parts for f in p[3]],
                              "source": source})
        frames.append(frame)
        fps.append(np.concatenate([p[4] for p in parts]))
        log(f"{source}: {len(smiles):,} SMILES -> {len(frame):,} structures in range "
            f"({frame.key.nunique():,} unique) in {time.time() - t0:.0f}s")
    if not frames:
        return None, None
    frame = pd.concat(frames, ignore_index=True)
    fp = np.concatenate(fps)
    keep = ~frame.key.duplicated().to_numpy()
    return frame[keep].reset_index(drop=True), fp[keep]


# ---------- PubChem, restricted to the query mass windows ----------


def find_pubchem(root="/kaggle/input"):
    """Paths of PubChem's CID-Mass and CID-SMILES files (gzipped or not), or None."""
    found = {}
    for f in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        name = os.path.basename(f).lower()
        if os.path.isfile(f) and name.startswith("cid-mass"):
            found["mass"] = f
        elif os.path.isfile(f) and name.startswith("cid-smiles"):
            found["smiles"] = f
    return found if len(found) == 2 else None


def _compression(path):
    with open(path, "rb") as f:
        return "gzip" if f.read(2) == b"\x1f\x8b" else None


ORGANIC = r"^(?:(?:C|H|N|O|P|S|F|Cl|Br|I)\d*)+$"  # neutral formulas made only of these elements


def pubchem_in_windows(mass_path, smiles_path, masses, ppm=10.0, cap=None, chunk=5_000_000, log=print):
    """SMILES of PubChem compounds within ppm of a query mass, at most `cap` per query.

    Keeps neutral, organic-element formulas. PubChem is dense (~8,000 compounds per typical 10 ppm window);
    cap=None keeps them all, an integer keeps the `cap` lowest CIDs (longest-known) per window.
    Streams both files once; adapts to whatever test set is attached.
    """
    q = np.unique(np.asarray(masses, dtype=float))
    q = q[np.isfinite(q)]
    if len(q) == 0:
        return np.array([], dtype=object)
    q = np.r_[q, q[-1]] if len(q) == 1 else q
    t0, cids, monos, n = time.time(), [], [], 0
    for df in pd.read_csv(mass_path, sep="\t", header=None, names=["cid", "formula", "mono", "exact"],
                          usecols=["cid", "formula", "mono"],
                          dtype={"cid": np.int64, "formula": str, "mono": np.float64},
                          chunksize=chunk, compression=_compression(mass_path), on_bad_lines="skip"):
        m = df.mono.to_numpy()
        i = np.searchsorted(q, m).clip(1, len(q) - 1)
        near = np.minimum(np.abs(m - q[i]), np.abs(m - q[i - 1])) <= m * ppm * 1e-6
        sub = df[near]
        sub = sub[sub.formula.str.fullmatch(ORGANIC, na=False)]
        cids.append(sub.cid.to_numpy())
        monos.append(sub.mono.to_numpy())
        n += len(df)
    cid, mono = np.concatenate(cids), np.concatenate(monos)
    order = np.argsort(mono, kind="stable")
    cid, mono = cid[order], mono[order]
    keep, truncated = [], 0
    for x in q:
        tol = x * ppm * 1e-6
        lo, hi = np.searchsorted(mono, x - tol), np.searchsorted(mono, x + tol, side="right")
        window = cid[lo:hi]
        if cap and len(window) > cap:
            truncated += 1
            window = np.partition(window, cap - 1)[:cap]
        keep.append(window)
    keep = np.unique(np.concatenate(keep))
    log(f"pubchem: {n:,} scanned, {len(cid):,} organic & neutral in windows; kept {len(keep):,} "
        f"(cap {cap}: {truncated} of {len(q)} windows truncated) ({time.time() - t0:.0f}s)")
    out = []
    for df in pd.read_csv(smiles_path, sep="\t", header=None, names=["cid", "smiles"],
                          dtype={"cid": np.int64, "smiles": str}, chunksize=chunk, quoting=3,
                          compression=_compression(smiles_path), on_bad_lines="skip"):
        out.append(df.smiles.to_numpy()[np.isin(df.cid.to_numpy(), keep)])
    smiles = np.concatenate(out) if out else np.array([], dtype=object)
    log(f"pubchem: {len(smiles):,} SMILES read ({time.time() - t0:.0f}s)")
    return smiles


# ---------- PubChem channel: documentation prior and a per-molecule shortlist ----------


def find_pubchem_docs(root="/kaggle/input"):
    """Paths of PubChem's CID-SID and CID-PMID files, or None."""
    found = {}
    for f in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        name = os.path.basename(f).lower()
        if os.path.isfile(f) and name.startswith("cid-sid"):
            found["sid"] = f
        elif os.path.isfile(f) and name.startswith("cid-pmid"):
            found["pmid"] = f
    return found if len(found) == 2 else None


def window_compounds(mass_path, masses, ppm=10.0, chunk=5_000_000):
    """(cid, mono mass) of neutral organic PubChem compounds within ppm of any query mass, sorted by mass."""
    q = np.unique(np.asarray(masses, dtype=float))
    q = q[np.isfinite(q)]
    q = np.r_[q, q[-1]] if len(q) == 1 else q
    cids, monos = [], []
    for df in pd.read_csv(mass_path, sep="\t", header=None, names=["cid", "formula", "mono", "exact"],
                          usecols=["cid", "formula", "mono"],
                          dtype={"cid": np.int64, "formula": str, "mono": np.float64},
                          chunksize=chunk, compression=_compression(mass_path), on_bad_lines="skip"):
        m = df.mono.to_numpy()
        i = np.searchsorted(q, m).clip(1, len(q) - 1)
        sub = df[np.minimum(np.abs(m - q[i]), np.abs(m - q[i - 1])) <= m * ppm * 1e-6]
        sub = sub[sub.formula.str.fullmatch(ORGANIC, na=False)]
        cids.append(sub.cid.to_numpy())
        monos.append(sub.mono.to_numpy())
    cid, mono = np.concatenate(cids), np.concatenate(monos)
    order = np.argsort(mono, kind="stable")
    return cid[order], mono[order]


def count_rows_per_cid(path, cids, chunk=10_000_000):
    """Number of rows per CID in a CID-keyed PubChem file (CID-SID, CID-PMID), aligned with `cids`."""
    uniq = np.unique(cids)
    counts = np.zeros(len(uniq), dtype=np.int64)
    for df in pd.read_csv(path, sep="\t", header=None, usecols=[0], dtype={0: np.int64}, chunksize=chunk,
                          compression=_compression(path), on_bad_lines="skip"):
        c = df[0].to_numpy()
        c = c[np.isin(c, uniq)]
        if len(c):
            v, n = np.unique(c, return_counts=True)
            counts[np.searchsorted(uniq, v)] += n
    return counts[np.searchsorted(uniq, cids)]


def documentation_prior(n_sid, n_pmid):
    """pop = ln(1 + substance records) + ln(1 + PubMed articles): how much a compound has been written about."""
    return np.log1p(n_sid) + np.log1p(n_pmid)


def shortlist_windows(cid, mono, pop, masses, ppm=10.0, depth=1000):
    """Per query mass: the `depth` best-documented compounds of its window (CIDs), and the union of all of them."""
    lists = []
    for x in masses:
        tol = x * ppm * 1e-6
        lo, hi = np.searchsorted(mono, x - tol), np.searchsorted(mono, x + tol, side="right")
        w = np.arange(lo, hi)
        if len(w) > depth:
            w = w[np.argsort(-pop[w], kind="stable")[:depth]]
        lists.append(cid[w])
    return lists, (np.unique(np.concatenate(lists)) if lists else np.array([], dtype=np.int64))


def smiles_for_cids(smiles_path, cids, chunk=5_000_000):
    """{cid: SMILES} for the given CIDs (one pass over CID-SMILES)."""
    want, out = np.unique(cids), {}
    for df in pd.read_csv(smiles_path, sep="\t", header=None, names=["cid", "smiles"],
                          dtype={"cid": np.int64, "smiles": str}, chunksize=chunk, quoting=3,
                          compression=_compression(smiles_path), on_bad_lines="skip"):
        hit = np.isin(df.cid.to_numpy(), want)
        out.update(zip(df.cid.to_numpy()[hit].tolist(), df.smiles.to_numpy()[hit].tolist(), strict=True))
    return out


def convert_cids(cid_smiles, n_bits=2048, workers=None, chunk=20000):
    """RDKit for {cid: SMILES} -> frame[cid, key, smiles, mass, formula] and packed fingerprints (row-aligned)."""
    cids = np.array(list(cid_smiles), dtype=np.int64)
    smiles = [str(cid_smiles[c]) for c in cids]
    jobs = [(smiles[i:i + chunk], n_bits) for i in range(0, len(smiles), chunk)]
    with get_context("fork").Pool(workers or os.cpu_count() or 1) as pool:
        parts = pool.map(_convert, jobs)
    rows, fps = [], []
    for j, p in enumerate(parts):
        base = j * chunk
        rows += [(int(cids[base + k]), key, smi, m, f) for k, key, smi, m, f in zip(p[5], *p[:4], strict=True)]
        fps.append(p[4])
    frame = pd.DataFrame(rows, columns=["cid", "key", "smiles", "mass", "formula"])
    return frame, (np.concatenate(fps) if fps else np.zeros((0, n_bits // 8), dtype=np.uint8))


def merge_channel(main, channel, slots=(), override=False, n=25):
    """Insert channel proposals into a main list at fixed final ranks (1-based `slots`); `override` puts the
    first proposal at rank 1 (the remaining proposals then take the slots). Items are pool/channel ids;
    duplicates are dropped and the list is cut to n."""
    channel = list(channel)
    out, used = [], set()
    take = []
    if override and channel:
        take.append((1, channel.pop(0)))
    take += [(s, c) for s, c in zip(sorted(slots), channel, strict=False)]
    insert_at = dict(take)
    main_iter = iter(main)
    for rank in range(1, n + 1):
        item = insert_at.get(rank)
        if item is None:
            item = next((m for m in main_iter if m not in used), None)
        if item is not None and item not in used:
            out.append(item)
            used.add(item)
    return out


# ---------- precursor mass ----------

WATER = 18.01056468


def consensus_mass(masses, ppm=10.0):
    """Neutral mass most spectra of a molecule agree on (within ppm), not the median.

    With an even number of spectra the median averages the middle two, so one spectrum with an
    unusual or mislabelled adduct can move the mass off every correct candidate.
    """
    m = np.asarray(masses, dtype=float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return float("nan")
    tol = float(np.median(m)) * ppm * 1e-6
    support = (np.abs(m[:, None] - m[None, :]) <= tol).sum(axis=1)
    tied = m[support == support.max()]
    centre = tied[np.argmin(np.abs(tied - np.median(m)))]
    return float(m[np.abs(m - centre) <= tol].mean())


# ---------- official-key deduplication of the submission ----------


def _official_keys(smiles_list):
    """Worker: InChIKey14 after RDKit tautomer canonicalisation (how the metric compares answers)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    enum = rdMolStandardize.TautomerEnumerator()
    enum.SetMaxTautomers(256)  # cap for huge natural products; dedup only, so an approximation is safe
    out = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            out.append(Chem.MolToInchiKey(enum.Canonicalize(mol))[:14] if mol is not None else s)
        except Exception:  # noqa: BLE001
            out.append(s)
    return out


def dedupe_official(rankings, pool, k=25, workers=None, log=print):
    """Drop guesses that the metric would treat as the same structure (tautomers), keep the first k.

    rankings: {molecule: [pool indices]} longer than k, so freed slots are refilled from further down.
    """
    smiles = pool.smiles.to_numpy()
    needed = sorted({int(i) for r in rankings.values() for i in r})
    chunks = [[smiles[i] for i in needed[j:j + 500]] for j in range(0, len(needed), 500)]
    with get_context("fork").Pool(workers or os.cpu_count() or 1) as p:
        keys = [key for part in p.map(_official_keys, chunks) for key in part]
    official = dict(zip(needed, keys, strict=True))
    out, freed = {}, 0
    for mol, ranked in rankings.items():
        seen, kept = set(), []
        for i in ranked[:k]:
            freed += official[int(i)] in seen
            seen.add(official[int(i)])
        seen = set()
        for i in ranked:
            if official[int(i)] not in seen:
                seen.add(official[int(i)])
                kept.append(int(i))
            if len(kept) == k:
                break
        out[mol] = kept
    log(f"official-key dedup: {freed} duplicate slots in the top {k} replaced ({len(needed):,} structures checked)")
    return out


# ---------- gated B2 fusion ----------


def candidate_lib_scores(cp, pool, library, spectra, spectra_mz, spectra_in, cands, modified=True):
    """{molecule: array}: for each candidate, mean over the molecule's spectra of its best library match.

    0 for candidates without reference spectra. cands: {molecule: [pool indices]}.
    """
    keys = pool.key.to_numpy()
    spectra = spectra.assign(pos=np.arange(len(spectra)))
    out = {}
    for mol, group in spectra.groupby("molecule_id", sort=False):
        if mol not in cands:
            continue
        rows = [(r.pos, r.ionization_mode, r.precursor_mz, int((spectra_in[r.pos] > 0).sum()))
                for r in group.itertuples()]
        vals = np.zeros(len(cands[mol]))
        for c, idx in enumerate(cands[mol]):
            per_spec = []
            for pos, mode, prec, n in rows:
                best = 0.0
                for lib_pos in library.index.get((keys[idx], mode), ()) if n else ():
                    m = library.n_peaks[lib_pos]
                    if m == 0:
                        continue
                    shift = (prec - library.precursor[lib_pos]) if modified else None
                    best = max(best, cp.match_score(spectra_mz[pos, :n], spectra_in[pos, :n],
                                                    library.mz[lib_pos, :m], library.intensity[lib_pos, :m],
                                                    shift=shift))
                per_spec.append(best)
            vals[c] = float(np.mean(per_spec)) if per_spec else 0.0
        out[mol] = vals
    return out


def b2_top_scores(cp, pool, library, spectra, spectra_mz, spectra_in, b2_rank, modified=True):
    """Per molecule: the library score of B2's top candidate (the number B2 ranked by, pooling="mean")."""
    top = {mol: [r[0]] for mol, r in b2_rank.items()}
    scores = candidate_lib_scores(cp, pool, library, spectra, spectra_mz, spectra_in, top, modified)
    return {mol: float(v[0]) for mol, v in scores.items()}


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
