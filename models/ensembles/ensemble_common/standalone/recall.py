"""v3: PubChem-scale candidate recall, documentation and natural-product priors, hidden-test-like tuning.

Why. v2 scored 0.687 on the team split but 0.200 on the leaderboard. On the split the answer is always a
training structure in our pool. On the hidden test most answers are natural products that are not training
structures, so the score is capped by recall (is the answer a candidate at all?) and by telling the real
structure apart from thousands of same-mass isomers. The public 0.45-0.51 notebooks win on exactly these two
points: a 105M-structure PubChem search and a documentation prior (well-recorded structures first).

What this module adds (ideas from the public CASMI 2026 notebooks, re-implemented here; data from public
Kaggle datasets, credited in the notebook):

  PubChemTier       the mass-sorted PubChem tier (SMILES + offsets + masses, with aligned documentation and
                    natural-product arrays), read memory-mapped
  tier_shortlists   per molecule: every tier structure in the +-ppm window (capped by documentation), Morgan
                    bits computed on the fly and scored with OUR fingerprint model; the best by log-likelihood,
                    by soft Tanimoto and by fingerprint + documentation are kept as extra candidates
  candidate_priors  per pool structure: pop = ln(1+PubChem substances) + ln(1+PubMed articles), and a
                    natural-product flag (COCONUT, LOTUS, NPAtlas)
  score_v3          v2's three sources plus "pop" and "np", over the pool window plus the molecule's shortlist
  novel copies      each validation molecule is scored a second time as if its structure were unknown: it is
                    removed from the training structures (kept only if COCONUT / PubChemLite / PubChem also hold
                    it) and its library spectra do not count. That is the hidden test's situation.
  tune_v3           v2's convex weight search, cross-fitted, on a hidden-like objective instead of balanced MRR
"""

from __future__ import annotations

import glob
import os
import time
import zlib
from multiprocessing import get_context

import numpy as np
import pandas as pd

from . import teamsplit as T

SOURCES = ("library", "fp_ll", "fp_tani", "pop", "np")
BASE_SOURCES = ("library", "fp_ll", "fp_tani")
# objective: share of weight per molecule group (standard class 2, unknown-structure copies)
OBJECTIVE = {"std_class2": 0.2, "novel_np": 0.4, "novel_other": 0.4}


# ---------- PubChem tier ----------


class PubChemTier:
    """Memory-mapped PubChem tier: ``pc_mass`` (sorted), ``pc_off`` (n+1 offsets), ``pc_smiles`` (bytes).

    Optional aligned arrays: ``pc_lsid`` = ln(1+substances), ``pc_lpmid`` = ln(1+PubMed), ``pc_np`` (0/1).
    """

    def __init__(self, folder: str, prior_folder: str | None = None):
        self.folder = folder
        self.mass = np.load(os.path.join(folder, "pc_mass.npy"), mmap_mode="r")
        self.off = np.load(os.path.join(folder, "pc_off.npy"), mmap_mode="r")
        self.buf = np.load(os.path.join(folder, "pc_smiles.npy"), mmap_mode="r")
        if len(self.off) != len(self.mass) + 1:
            raise ValueError(f"pc_off has {len(self.off)} rows for {len(self.mass)} masses")
        self.arrays = {}
        for name in ("pc_lsid", "pc_lpmid", "pc_np"):
            for d in (prior_folder, folder, os.path.dirname(folder)):
                p = os.path.join(d, f"{name}.npy") if d else ""
                if p and os.path.exists(p):
                    a = np.load(p, mmap_mode="r")
                    if len(a) == len(self.mass):
                        self.arrays[name] = a
                        break

    @classmethod
    def find(cls, root: str = "/kaggle/input") -> PubChemTier | None:
        hits = sorted(glob.glob(os.path.join(root, "**", "pc_smiles.npy"), recursive=True), key=len)
        if not hits:
            return None
        folder = os.path.dirname(hits[0])
        priors = sorted(glob.glob(os.path.join(root, "**", "pc_lsid.npy"), recursive=True), key=len)
        return cls(folder, os.path.dirname(priors[0]) if priors else None)

    def __len__(self) -> int:
        return len(self.mass)

    def bounds(self, mass: float, ppm: float) -> tuple[int, int]:
        tol = mass * ppm * 1e-6
        return int(np.searchsorted(self.mass, mass - tol)), int(np.searchsorted(self.mass, mass + tol, side="right"))

    def smiles(self, rows) -> list[str]:
        """SMILES of ``rows``, read in file order (the arrays are large memory maps)."""
        rows = np.asarray(rows, dtype=np.int64)
        off, buf = self.off, self.buf
        out = [""] * len(rows)
        for k in np.argsort(rows, kind="stable"):
            r = rows[k]
            out[k] = bytes(buf[off[r] : off[r + 1]]).decode()
        return out

    def take(self, name: str, rows) -> np.ndarray:
        """``self.<name>[rows]`` read in file order (``name``: "mass" or one of ``self.arrays``)."""
        rows = np.asarray(rows, dtype=np.int64)
        arr = self.mass if name == "mass" else self.arrays[name]
        order = np.argsort(rows, kind="stable")
        out = np.empty(len(rows), dtype=arr.dtype)
        out[order] = arr[rows[order]]
        return out

    def prior(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.zeros(len(rows), dtype=np.float32)
        for name in ("pc_lsid", "pc_lpmid"):
            if name in self.arrays:
                out += self.take(name, rows).astype(np.float32)
        return out

    def is_np(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if "pc_np" not in self.arrays:
            return np.zeros(len(rows), dtype=np.uint8)
        return (self.take("pc_np", rows) != 0).astype(np.uint8)


_TIER: PubChemTier | None = None   # set in the parent before forking, so workers share the memory maps
_GEN = {}


def _generator(n_bits: int):
    if n_bits not in _GEN:
        from rdkit import RDLogger
        from rdkit.Chem import rdFingerprintGenerator

        RDLogger.DisableLog("rdApp.*")
        _GEN[n_bits] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return _GEN[n_bits]


def _top(x: np.ndarray, k: int) -> np.ndarray:
    ok = np.flatnonzero(np.isfinite(x))
    if len(ok) <= k:
        return ok
    return ok[np.argpartition(-x[ok], k - 1)[:k]]


def _shortlist(task):
    """Worker: one molecule's tier window -> (i, tier rows, keys, packed bits, window size)."""
    from rdkit import Chem

    i, mass, pr, opt = task
    tier = _TIER
    n_bits = len(pr)
    empty = (i, np.zeros(0, np.int64), [], np.zeros((0, n_bits // 8), np.uint8), 0)
    if tier is None or not np.isfinite(mass):
        return empty
    lo, hi = tier.bounds(mass, opt["ppm"])
    if hi <= lo:
        return empty
    rows = np.arange(lo, hi, dtype=np.int64)
    prior = tier.prior(rows)
    if len(rows) > opt["w_max"]:            # keep the best documented, as the public channel's shortlist does
        keep = np.sort(np.argsort(-prior, kind="stable")[: opt["w_max"]])
        rows, prior = rows[keep], prior[keep]
    gen = _generator(n_bits)
    smis = tier.smiles(rows)
    bits = np.zeros((len(rows), n_bits), dtype=np.uint8)
    ok = np.zeros(len(rows), dtype=bool)
    for j, s in enumerate(smis):
        try:
            mol = Chem.MolFromSmiles(s)
        except Exception:  # noqa: BLE001
            mol = None
        if mol is not None:
            bits[j] = gen.GetFingerprintAsNumPy(mol)
            ok[j] = True
    pr = np.clip(np.asarray(pr, np.float64), 1e-4, 1 - 1e-4)
    z = np.log(pr / (1 - pr)).astype(np.float32)
    bf = bits.astype(np.float32)
    ll = np.where(ok, bf @ z, -np.inf)
    overlap = bf @ pr.astype(np.float32)
    tani = np.where(ok, overlap / (pr.sum() + bf.sum(axis=1) - overlap + 1e-8), -np.inf)
    zl = np.full(len(rows), -np.inf)
    if ok.any():
        m, s = ll[ok].mean(), ll[ok].std()
        zl[ok] = (ll[ok] - m) / (s if s > 1e-9 else 1.0)
    mix = np.where(ok, zl + opt["pop_w"] * prior, -np.inf)
    sel = np.unique(np.concatenate([_top(ll, opt["k_fp"]), _top(tani, opt["k_fp"]), _top(mix, opt["k_mix"])]))
    keep, keys = [], []
    for j in sel:
        mol = Chem.MolFromSmiles(smis[j])
        key = Chem.MolToInchiKey(mol)[:14] if mol is not None else ""
        if len(key) == 14:
            keep.append(j)
            keys.append(key)
    keep = np.asarray(keep, dtype=np.int64)
    return i, rows[keep], keys, np.packbits(bits[keep], axis=1), hi - lo


def tier_shortlists(tier: PubChemTier | None, mols, masses, probs_mol: np.ndarray, ppm: float = 10.0,
                    w_max: int = 20_000, k_fp: int = 60, k_mix: int = 60, pop_w: float = 0.25,
                    workers: int | None = None, max_minutes: float | None = None, batch: int = 64, log=print):
    """Shortlist tier structures for each molecule. Returns (hits, frame).

    hits: {molecule: array of tier rows}. frame: one row per unique tier row with key, packed fp.
    Molecules are processed in the given order in batches; when ``max_minutes`` is reached the rest get none.
    """
    global _TIER
    _TIER = tier
    mols = list(mols)
    hits, rows_all, keys_all, fps_all, sizes = {}, [], [], [], []
    if tier is None or not mols:
        return hits, pd.DataFrame({"tier_row": [], "key": []}), np.zeros((0, probs_mol.shape[1] // 8), np.uint8)
    opt = {"ppm": ppm, "w_max": w_max, "k_fp": k_fp, "k_mix": k_mix, "pop_w": pop_w}
    tasks = [(i, float(masses[i]), probs_mol[i], opt) for i in range(len(mols))]
    workers = workers or os.cpu_count() or 1
    t0 = time.time()
    pool = get_context("fork").Pool(workers) if workers > 1 else None
    try:
        for b in range(0, len(tasks), batch):
            if max_minutes is not None and (time.time() - t0) / 60 > max_minutes:
                log(f"  tier: time budget reached after {b:,} of {len(tasks):,} molecules")
                break
            part = tasks[b : b + batch]
            out = pool.map(_shortlist, part, chunksize=1) if pool else [_shortlist(t) for t in part]
            for i, rows, keys, fp, n in out:
                hits[mols[i]] = rows
                rows_all.append(rows)
                keys_all += keys
                fps_all.append(fp)
                sizes.append(n)
            if (b // batch) % 10 == 0:
                log(f"  tier: {min(b + batch, len(tasks)):,}/{len(tasks):,} molecules, "
                    f"median window {int(np.median(sizes)) if sizes else 0:,}")
    finally:
        if pool:
            pool.close()
            pool.join()
    frame = pd.DataFrame({"tier_row": np.concatenate(rows_all) if rows_all else np.zeros(0, np.int64),
                          "key": keys_all})
    fp = np.concatenate(fps_all) if fps_all else np.zeros((0, probs_mol.shape[1] // 8), np.uint8)
    first = ~frame.tier_row.duplicated().to_numpy()
    frame, fp = frame[first].reset_index(drop=True), fp[first]
    log(f"tier: {len(hits):,} molecules, {len(frame):,} unique structures, window sizes median "
        f"{int(np.median(sizes)) if sizes else 0:,} max {max(sizes) if sizes else 0:,}, "
        f"{(time.time() - t0) / 60:.1f} min")
    return hits, frame, fp


def extend_pool(pool: pd.DataFrame, pool_fp: np.ndarray, tier: PubChemTier | None, hits: dict,
                frame: pd.DataFrame, frame_fp: np.ndarray):
    """Append shortlisted tier structures to the (mass-sorted) pool, after its last row.

    Returns (pool_all, fp_all, mol_idx): ``mol_idx[mol]`` are pool_all rows of that molecule's shortlist.
    The first ``len(pool)`` rows are the original pool, unchanged, so libraries and windows still apply.
    One row per InChIKey first block; keys already in the pool point to the pool row.
    """
    n_main = len(pool)
    if tier is None or not len(frame):
        return pool.assign(tier_row=-1), pool_fp, {}
    main_key = pd.Series(np.arange(n_main), index=pool.key.to_numpy())
    main_key = main_key[~main_key.index.duplicated()]
    new = frame[~frame.key.isin(main_key.index)]
    new = new[~new.key.duplicated()]
    add = pd.DataFrame({"key": new.key.to_numpy(), "smiles": tier.smiles(new.tier_row.to_numpy()),
                        "formula": "", "mass": tier.take("mass", new.tier_row.to_numpy()).astype(float),
                        "n_ref": 0, "tier_row": new.tier_row.to_numpy()})
    cols = [c for c in pool.columns if c in add.columns] + ["tier_row"]
    pool_all = pd.concat([pool.assign(tier_row=-1), add[cols]], ignore_index=True)
    fp_all = np.concatenate([pool_fp, frame_fp[new.index.to_numpy()]])
    where = pd.concat([main_key, pd.Series(n_main + np.arange(len(add)), index=add.key.to_numpy())])
    row_key = pd.Series(frame.key.to_numpy(), index=frame.tier_row.to_numpy())
    mol_idx = {}
    for mol, rows in hits.items():
        keys = row_key.reindex(rows).dropna().to_numpy()
        mol_idx[mol] = np.unique(where.reindex(keys).dropna().to_numpy().astype(np.int64))
    return pool_all, fp_all, mol_idx


def molecule_probs(spectra: pd.DataFrame, probs: np.ndarray, mols) -> np.ndarray:
    """Mean predicted bit probabilities per molecule, in the order of ``mols``."""
    groups = spectra.reset_index(drop=True).groupby("molecule_id", sort=False).indices
    return np.stack([probs[groups[m]].astype(np.float32).mean(axis=0) for m in mols])


# ---------- priors ----------


def _read_cols(path: str, wanted) -> pd.DataFrame:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        names = pq.ParquetFile(path).schema.names
        return pd.read_parquet(path, columns=[c for c in wanted if c in names])
    head = pd.read_csv(path, nrows=0)
    return pd.read_csv(path, usecols=[c for c in wanted if c in head.columns], low_memory=False)


def candidate_priors(pool: pd.DataFrame, coconut_keys=(), popularity: str | None = None,
                     np_table: str | None = None, pubchemlite=(), tier: PubChemTier | None = None,
                     tier_rows: np.ndarray | None = None, log=print):
    """pop (float32) and np flag (uint8) for every pool row.

    pop = ln(1+substances) + ln(1+PubMed) from ``pool_popularity.csv`` (same scale as the tier arrays); for
    structures it does not cover, PubChemLite's ln(1+PubMed) + ln(1+patents) stands in. Tier rows take the
    tier arrays (``tier_rows`` gives the tier row of each pool row, -1 for non-tier rows).
    np = COCONUT, LOTUS or NPAtlas membership.
    """
    keys = pool.key.to_numpy()
    pop = np.full(len(pool), np.nan, dtype=np.float32)
    # pandas isin is hash based; np.isin on object (string) arrays compares every pair, which is hours here
    npf = pd.Index(keys).isin(list(coconut_keys))
    if popularity:
        p = _read_cols(popularity, ["inchikey14", "log1p_substances", "log1p_pubmed", "np_coconut", "np_lotus",
                                    "np_npatlas"]).drop_duplicates("inchikey14")
        s = p.set_index("inchikey14")
        v = (s.get("log1p_substances", 0) + s.get("log1p_pubmed", 0)).astype(np.float32)
        hit = pd.Series(keys).map(v).to_numpy(dtype=np.float32)
        pop = np.where(np.isnan(pop), hit, pop)
        flag = np.zeros(len(s), dtype=bool)
        for c in ("np_coconut", "np_lotus", "np_npatlas"):
            if c in s:
                flag |= s[c].fillna(0).to_numpy() > 0
        npf |= pd.Series(keys).map(pd.Series(flag, index=s.index)).fillna(False).to_numpy(dtype=bool)
        log(f"priors: pool_popularity covers {np.isfinite(hit).mean():.1%} of the pool")
    for path in pubchemlite:
        p = _read_cols(path, ["FirstBlock", "PubMed_Count", "Patent_Count"]).dropna(subset=["FirstBlock"])
        p = p.drop_duplicates("FirstBlock").set_index("FirstBlock")
        v = (np.log1p(p.get("PubMed_Count", 0).fillna(0)) + np.log1p(p.get("Patent_Count", 0).fillna(0)))
        hit = pd.Series(keys).map(v.astype(np.float32)).to_numpy(dtype=np.float32)
        pop = np.where(np.isnan(pop), hit, pop)
    if np_table:
        t = _read_cols(np_table, ["ik14"])
        npf |= pd.Index(keys).isin(t.ik14.dropna().unique())
    if tier is not None and tier_rows is not None:
        m = tier_rows >= 0
        if m.any():
            pop[m] = tier.prior(tier_rows[m])
            npf[m] |= tier.is_np(tier_rows[m]).astype(bool)
    pop = np.nan_to_num(pop, nan=0.0)
    log(f"priors: mean pop {pop.mean():.2f}, natural-product flag on {npf.mean():.1%} of the pool")
    return pop.astype(np.float32), npf.astype(np.uint8)


# ---------- scoring ----------


def score_v3(cp, spectra: pd.DataFrame, X: np.ndarray, probs: np.ndarray, cent: T.Centroids,
             pool_all: pd.DataFrame, fp_all: np.ndarray, pop: np.ndarray, npf: np.ndarray, n_main: int,
             cfg: T.FusionConfig, mols=None, extra_idx: dict | None = None, truth_keys: dict | None = None,
             novel: bool = False, ext_keys=frozenset()) -> T.Scores:
    """v2's ``score_molecules`` over the pool window plus each molecule's shortlist, with pop and np.

    novel=True scores every molecule as if its structure were unknown: its pool row is removed unless an
    external source (``ext_keys``: COCONUT / PubChemLite) also holds it, and it gets no library score.
    """
    keys = pool_all.key.to_numpy()
    pool_mass = pool_all.mass.to_numpy()[:n_main]
    modes = T.mode_ids(spectra.ionization_mode)
    nmass = spectra.neutral_mass.to_numpy(dtype=float)
    groups = spectra.reset_index(drop=True).groupby("molecule_id", sort=False).indices
    mols = list(pd.unique(spectra.molecule_id)) if mols is None else list(mols)
    extra_idx = extra_idx or {}
    segs, cands = [], []
    conf, truth, masses = np.zeros(len(mols)), np.full(len(mols), -1), np.zeros(len(mols))
    src = {s: [] for s in SOURCES}
    flat = 0
    for i, mol in enumerate(mols):
        p = groups[mol]
        mass = float(np.nanmedian(nmass[p]))
        masses[i] = mass
        idx = cp.window_indices(pool_mass, mass, cfg.ppm) if np.isfinite(mass) else np.zeros(0, dtype=np.int64)
        if len(idx) > cfg.max_candidates:
            idx = np.sort(idx[np.argsort(np.abs(pool_mass[idx] - mass), kind="stable")[: cfg.max_candidates]])
        tk = truth_keys.get(mol) if truth_keys is not None else None
        if novel and tk is not None and tk not in ext_keys:
            idx = idx[keys[idx] != tk]
        ext = np.asarray(extra_idx.get(mol, ()), dtype=np.int64)
        if len(ext):
            ext = ext[~np.isin(ext, idx)]
        n_win = len(idx)
        idx = np.concatenate([idx, ext]).astype(np.int64)
        if not len(idx):
            continue
        lib = np.zeros(len(idx), dtype=np.float32)
        main = idx[:n_win]
        for m in np.unique(modes[p]):
            lib[:n_win] += cent.scores(main, int(m), X[p[modes[p] == m]])
        lib /= len(p)
        if novel and tk is not None:
            lib[keys[idx] == tk] = 0.0
        pr = np.clip(probs[p].astype(np.float64).mean(axis=0), 1e-4, 1 - 1e-4)
        bits = np.unpackbits(fp_all[idx], axis=1)[:, : len(pr)].astype(np.float32)
        overlap = bits @ pr.astype(np.float32)
        src["library"].append(lib)
        src["fp_ll"].append(bits @ np.log(pr / (1 - pr)).astype(np.float32))
        src["fp_tani"].append(overlap / (pr.sum() + bits.sum(axis=1) - overlap + 1e-8))
        src["pop"].append(pop[idx].astype(np.float32))
        src["np"].append(npf[idx].astype(np.float32))
        segs.append(np.full(len(idx), i))
        cands.append(idx)
        conf[i] = float(lib.max())
        if tk is not None:
            hit = np.flatnonzero(keys[idx] == tk)
            truth[i] = flat + int(hit[0]) if len(hit) else -1
        flat += len(idx)

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    return T.Scores(mols=mols, mass=masses, seg=cat(segs, np.int64), cand=cat(cands, np.int64),
                    src={s: cat(v, np.float64) for s, v in src.items()}, conf=conf,
                    truth=truth if truth_keys is not None else None)


def rankings_v3(cp, sc: T.Scores, f: np.ndarray, pool_all: pd.DataFrame, n_main: int,
                exclude: dict | None = None) -> dict:
    """molecule -> pool_all rows, best first, filled to 25 with mass neighbours from the main pool.

    ``exclude``: {molecule: key} never used as filler (the removed structure in novel copies).
    """
    keys = pool_all.key.to_numpy()
    pool_mass = pool_all.mass.to_numpy()[:n_main]
    off = sc.offsets
    out = {}
    for i, mol in enumerate(sc.mols):
        idx, fi = sc.cand[off[i] : off[i + 1]], f[off[i] : off[i + 1]]
        ranked = list(dict.fromkeys(int(j) for j in idx[np.lexsort((idx, -fi))][: cp.TOP_K])) if len(idx) else []
        bad = (exclude or {}).get(mol)
        if len(ranked) < cp.TOP_K and np.isfinite(sc.mass[i]):
            seen = set(ranked)
            for j in cp.nearest_indices(pool_mass, sc.mass[i], 4 * cp.TOP_K):
                j = int(j)
                if j not in seen and keys[j] != bad:
                    ranked.append(j)
                    seen.add(j)
                if len(ranked) == cp.TOP_K:
                    break
        out[mol] = ranked
    return out


# ---------- tuning on a hidden-like objective ----------


def stack(a: T.Scores, b: T.Scores) -> T.Scores:
    """Both molecule sets in one Scores (b's molecules follow a's)."""
    n, f = len(a.mols), len(a.seg)
    return T.Scores(mols=list(a.mols) + list(b.mols), mass=np.r_[a.mass, b.mass], seg=np.r_[a.seg, b.seg + n],
                    cand=np.r_[a.cand, b.cand], src={s: np.r_[a.src[s], b.src[s]] for s in a.src},
                    conf=np.r_[a.conf, b.conf], truth=np.r_[a.truth, np.where(b.truth >= 0, b.truth + f, -1)])


def objective_weights(cls: np.ndarray, is_np: np.ndarray, n_std: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Per molecule copy (standard copies first, then novel copies) weights of the hidden-like objective."""
    n = len(cls)
    groups = {"std_class2": np.r_[cls == 2, np.zeros(n, bool)],
              "novel_np": np.r_[np.zeros(n, bool), is_np],
              "novel_other": np.r_[np.zeros(n, bool), ~is_np]}
    assert n_std == n
    mask = np.ones(2 * n, bool) if mask is None else np.r_[mask, mask]
    w = np.zeros(2 * n)
    for g, share in OBJECTIVE.items():
        m = groups[g] & mask
        if m.any():
            w[m] = share / m.sum()
    return w / max(w.sum(), 1e-12)


def source_matrix(sc: T.Scores, sources, method: str, cfg: T.FusionConfig) -> np.ndarray:
    return np.stack([T.normalize(sc.src[s], sc.seg, len(sc.mols), method, cfg.rrf_k) for s in sources], axis=1)


def fit_weighted(kind: str, N: np.ndarray, sc: T.Scores, w_mol: np.ndarray, cfg: T.FusionConfig) -> dict:
    grid = T.weight_grid(N.shape[1], cfg.n_random, cfg.seed)
    w_global, _ = T._best(N, sc, w_mol, grid)
    if kind == "global":
        return {"kind": kind, "weights": w_global}
    m0 = w_mol > 0
    pos = sc.conf[m0 & (sc.conf > 0)]
    qs = np.linspace(0, 1, cfg.n_conf_bins + 1)[1:-1]
    edges = np.unique(np.quantile(pos, qs)) if len(pos) else np.zeros(0)
    bins = T.conf_bins(sc.conf, edges)
    W = np.tile(w_global, (len(edges) + 2, 1))
    sizes = []
    for b in range(len(edges) + 2):
        m = m0 & (bins == b)
        sizes.append(int(m.sum()))
        if m.sum() >= cfg.min_bin:
            W[b], _ = T._best(N, sc, w_mol * m, grid)
    return {"kind": kind, "weights": w_global, "edges": edges, "bin_weights": W, "bin_sizes": sizes}


def tune_v3(cp, sc_std: T.Scores, sc_nov: T.Scores, truth: pd.DataFrame, pool_all: pd.DataFrame, n_main: int,
            cfg: T.FusionConfig, ext_keys=frozenset(), log=print) -> dict:
    """Singles, fusion over the v2 sources and over all five sources; cross-fitted; pick by the objective."""
    truth = truth.set_index("molecule_id").loc[sc_std.mols].reset_index()
    cls = truth.cls.to_numpy()
    is_np = (truth.query_lib == "enveda-np-examples").to_numpy()
    keys = pool_all.key.to_numpy()
    both = stack(sc_std, sc_nov)
    n = len(sc_std.mols)
    fold1 = T.folds(sc_std.mols)
    fold = np.r_[fold1, fold1]
    exclude = {m: k for m, k in zip(truth.molecule_id, truth.true_key, strict=True) if k not in ext_keys}

    def evaluate(f_both):
        f_std, f_nov = f_both[: len(sc_std.seg)], f_both[len(sc_std.seg) :]
        t_std = cp.evaluate(rankings_v3(cp, sc_std, f_std, pool_all, n_main), truth, keys, ci=True)
        t_nov = cp.evaluate(rankings_v3(cp, sc_nov, f_nov, pool_all, n_main, exclude), truth, keys)
        rr = np.r_[T.reciprocal_ranks(f_std, sc_std), T.reciprocal_ranks(f_nov, sc_nov)]
        obj = float((objective_weights(cls, is_np, n) * rr).sum())
        nov_rr = rr[n:]
        row = {"objective": round(obj, 4), "novel all": round(float(nov_rr.mean()), 4),
               "novel np": round(float(nov_rr[is_np].mean()), 4) if is_np.any() else np.nan,
               "balanced": t_std.loc["balanced", "MRR@25"], "class 1": t_std.loc["class 1", "MRR@25"],
               "class 2": t_std.loc["class 2", "MRR@25"], "np proxy": t_std.loc["np proxy", "MRR@25"],
               "top1": t_std.loc["balanced", "top1"]}
        return row, t_std, t_nov

    rows, params, tables, val_scores = [], {}, {}, {}
    for s in SOURCES:
        row, t_std, t_nov = evaluate(both.src[s])
        rows.append({"method": s, **row, "kind": "single"})
        params[s] = {"kind": "single", "source": s}
        tables[s], tables[f"{s} (novel)"] = t_std, t_nov
    for label, sources in (("v2", BASE_SOURCES), ("v3", SOURCES)):
        for method in ("z", "rrf"):
            N = source_matrix(both, sources, method, cfg)
            for kind in ("global", "gated"):
                if method == "rrf" and kind == "gated":
                    continue
                name = f"{label}_{kind}_{method}"
                full = fit_weighted(kind, N, both, objective_weights(cls, is_np, n), cfg)
                f_cf = np.zeros(len(both.seg))
                for k in (0, 1):
                    p = fit_weighted(kind, N, both, objective_weights(cls, is_np, n, fold1 == k), cfg)
                    rows_k = fold[both.seg] != k
                    f_cf[rows_k] = T.apply_variant(p, N, both)[rows_k]
                row, t_std, t_nov = evaluate(f_cf)
                in_obj = float((objective_weights(cls, is_np, n)
                                * np.r_[T.reciprocal_ranks(T.apply_variant(full, N, both)[: len(sc_std.seg)], sc_std),
                                        T.reciprocal_ranks(T.apply_variant(full, N, both)[len(sc_std.seg):],
                                                           sc_nov)]).sum())
                rows.append({"method": name, **row, "kind": "fusion (cross-fitted)", "in_sample": round(in_obj, 4)})
                params[name] = {**full, "method": method, "sources": list(sources)}
                tables[name], tables[f"{name} (novel)"] = t_std, t_nov
                val_scores[name] = f_cf[: len(sc_std.seg)]
                log(f"{name}: objective {row['objective']:.4f} (in-sample {in_obj:.4f}) | novel {row['novel all']:.4f}"
                    f" | balanced {row['balanced']:.4f} | weights "
                    f"{dict(zip(sources, np.round(full['weights'], 3).tolist(), strict=True))}")
    summary = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return {"summary": summary, "best": summary.method[0], "params": params, "tables": tables,
            "val_scores": val_scores, "exclude": exclude}


def apply_v3(report: dict, sc: T.Scores, cfg: T.FusionConfig, name: str | None = None) -> np.ndarray:
    p = report["params"][name or report["best"]]
    if p["kind"] == "single":
        return sc.src[p["source"]]
    return T.apply_variant(p, source_matrix(sc, p["sources"], p["method"], cfg), sc)


def unique_leaders(ranks: dict, f_top: dict) -> dict:
    """Every test molecule is a different structure: a structure leading several lists stays first only in the
    list where it scores highest and drops to rank 2 elsewhere. ``f_top``: molecule -> fused score of its leader.
    """
    lead = {}
    for m, r in ranks.items():
        if r:
            lead.setdefault(r[0], []).append(m)
    out = dict(ranks)
    for ms in lead.values():
        if len(ms) < 2:
            continue
        winner = max(ms, key=lambda m: f_top.get(m, -np.inf))
        for m in ms:
            if m != winner and len(out[m]) > 1:
                r = list(out[m])
                r[0], r[1] = r[1], r[0]
                out[m] = r
    return out


def describe(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    if "weights" in out and "sources" in params:
        out["weights"] = dict(zip(params["sources"], np.round(params["weights"], 4).tolist(), strict=True))
    return out


def crc_fold(m) -> int:
    return zlib.crc32(str(m).encode()) % 2
