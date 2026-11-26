"""Late fusion on the team validation split (Member 3's ``make_split``, used the same way as Member 7).

``cp`` is Member 3's ``casmi_pipeline`` module. It is passed in as an argument and never imported
here, so this package does not import from another member's folder. ``cp`` supplies the split, the
candidate pool, the mass window, ``fill_to_k``, ``evaluate`` (per-class MRR@25) and
``write_submission``, so every number is directly comparable with B1/B2 and Member 7's runs.

Sources, all trained inside the notebook:
  library : cosine between each query spectrum and the centroid of a candidate's library spectra in
            the same ion mode, averaged over the molecule's spectra. Validation uses only the rows in
            ``lib_mask`` (the split's library). The test uses every good-label training spectrum.
  fp_ll   : log-likelihood of the candidate's Morgan bits under the predicted bit probabilities.
  fp_tani : soft Tanimoto between the predicted probabilities and the candidate's bits.

Fusion: per-molecule normalisation (z-score or reciprocal rank), then convex weights from a random
Dirichlet search on balanced MRR@25. "gated" fits one weight set per library-confidence bin (best
library cosine in the window), because the library wins when references exist and the fingerprint
wins when they do not. Tuned variants are reported 2-fold cross-fitted: each molecule is scored with
weights tuned on the other half, so the reported number is never tuned on the molecules it scores.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SPEC_COLS = ("ms2_mzs", "ms2_normalized_intensities", "precursor_mz")
SOURCES = ("library", "fp_ll", "fp_tani")
METHODS = ("z", "rrf")
KINDS = ("global", "gated")


@dataclass
class FusionConfig:
    ppm: float = 10.0               # same window as Member 3 / Member 7
    n_bins: int = 1500              # 1 Da bins up to 1500 m/z
    bin_width: float = 1.0
    chunk_rows: int = 20_000        # rows binned at a time while streaming train.parquet
    max_candidates: int = 5000      # per molecule, nearest in mass (windows are rarely this big)
    n_random: int = 300             # Dirichlet weight draws per tuning
    n_conf_bins: int = 3            # gated: bins over positive library confidence (+ one bin for 0)
    min_bin: int = 40               # gated: smaller bins fall back to the global weights
    rrf_k: float = 60.0
    seed: int = 2026


# ---------- spectra -> unit vectors ----------


def mode_ids(values) -> np.ndarray:
    """1 for negative ion mode, 0 otherwise."""
    s = pd.Series(np.asarray(values, dtype=object)).astype(str).str.lower()
    return s.str.startswith("neg").to_numpy().astype(np.int64)


def side_features(cp, frame: pd.DataFrame) -> np.ndarray:
    """precursor m/z / 1000, ion mode and an adduct one-hot (index 0 = other adducts)."""
    n = len(frame)
    out = np.zeros((n, 2 + cp.N_ADDUCTS), dtype=np.float32)
    out[:, 0] = frame["precursor_mz"].to_numpy(dtype=float) / 1000.0
    out[:, 1] = mode_ids(frame["ionization_mode"])
    out[np.arange(n), 2 + cp.adduct_ids(frame["adduct"].astype(str))] = 1.0
    return out


def _flat(col, rows: np.ndarray):
    """Peak values of the selected list rows and the local row number of each peak."""
    off = np.asarray(col.offsets)
    vals = col.values.to_numpy(zero_copy_only=False)
    lo = off[rows]
    lens = off[rows + 1] - lo
    owner = np.repeat(np.arange(len(rows)), lens)
    pos = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(lo, lens)
    return vals[pos].astype(np.float64), owner


def bin_batch(batch, rows: np.ndarray, n_bins: int, bin_width: float) -> np.ndarray:
    """Binned, sqrt-scaled, L2-normalised spectra for ``rows`` of a record batch.

    Peaks within 0.5 Da of the precursor or above it are dropped (they are the same for every
    candidate in the mass window). A spectrum left with no peaks keeps them instead.
    """
    mz, owner = _flat(batch.column(SPEC_COLS[0]), rows)
    it, owner_i = _flat(batch.column(SPEC_COLS[1]), rows)
    if len(mz) != len(it) or not np.array_equal(owner, owner_i):
        raise ValueError("m/z and intensity lists have different lengths")
    prec = batch.column(SPEC_COLS[2]).to_numpy(zero_copy_only=False)[rows].astype(float)
    base = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (it > 0) & (mz < n_bins * bin_width)
    limit = np.where(np.isfinite(prec), prec - 0.5, np.inf)[owner]
    ok = base & (mz < limit)
    empty = np.bincount(owner[ok], minlength=len(rows)) == 0
    if empty.any():
        ok |= base & empty[owner]
    flat = owner[ok] * n_bins + (mz[ok] / bin_width).astype(np.int64)
    X = np.bincount(flat, weights=np.sqrt(it[ok]), minlength=len(rows) * n_bins)
    X = X.reshape(len(rows), n_bins).astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def iter_binned(path: str, select: np.ndarray | None, cfg: FusionConfig):
    """Yield (row_ids, X) over a spectra parquet in file order. ``select``: bool mask over rows."""
    f = pq.ParquetFile(path)
    start = 0
    for batch in f.iter_batches(batch_size=cfg.chunk_rows, columns=list(SPEC_COLS)):
        n = batch.num_rows
        local = np.arange(n) if select is None else np.flatnonzero(select[start : start + n])
        if len(local):
            yield start + local, bin_batch(batch, local, cfg.n_bins, cfg.bin_width)
        start += n


def bin_file(path: str, n_rows: int, cfg: FusionConfig) -> np.ndarray:
    """All rows of a (small) parquet, e.g. test.parquet, as float16."""
    out = np.zeros((n_rows, cfg.n_bins), dtype=np.float16)
    for rows, X in iter_binned(path, None, cfg):
        out[rows] = X
    return out


# ---------- centroid library ----------


class Centroids:
    """Unit-normalised summed spectrum per (structure, ion mode), for a chosen set of pool structures."""

    def __init__(self, pool_idx: np.ndarray, n_pool: int, n_bins: int):
        pool_idx = np.unique(np.asarray(pool_idx, dtype=np.int64))
        self.slot = np.full(n_pool, -1, dtype=np.int64)
        self.slot[pool_idx] = np.arange(len(pool_idx))
        self.sums = np.zeros((2 * len(pool_idx), n_bins), dtype=np.float32)
        self.counts = np.zeros(2 * len(pool_idx), dtype=np.int64)

    def ids(self, pool_idx: np.ndarray, mode) -> np.ndarray:
        """Centroid row for each pool index (-1 = no centroid kept for that structure)."""
        pool_idx = np.asarray(pool_idx, dtype=np.int64)
        s = np.where(pool_idx >= 0, self.slot[np.maximum(pool_idx, 0)], -1)
        return np.where(s >= 0, 2 * s + mode, -1)

    def add(self, cid: np.ndarray, X: np.ndarray) -> None:
        if not len(cid):
            return
        order = np.argsort(cid, kind="stable")
        c = cid[order]
        starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
        self.sums[c[starts]] += np.add.reduceat(X[order], starts, axis=0)
        self.counts[c[starts]] += np.diff(np.r_[starts, len(c)])

    def finish(self) -> None:
        self.sums /= np.maximum(np.linalg.norm(self.sums, axis=1, keepdims=True), 1e-12)

    def scores(self, pool_idx: np.ndarray, mode: int, Q: np.ndarray) -> np.ndarray:
        """Sum over the query spectra ``Q`` (same ion mode) of cosine to each candidate's centroid."""
        cid = self.ids(pool_idx, mode)
        ok = cid >= 0
        ok[ok] = self.counts[cid[ok]] > 0
        out = np.zeros(len(pool_idx), dtype=np.float32)
        if ok.any():
            out[ok] = (self.sums[cid[ok]] @ Q.astype(np.float32).T).sum(axis=1)
        return out


@dataclass
class Libraries:
    val: Centroids          # lib_mask rows only, structures in validation windows
    test: Centroids         # every good-label row, structures in test windows
    val_cid: np.ndarray     # per train.parquet row: centroid row in ``val`` or -1
    test_cid: np.ndarray


def window_union(cp, pool_mass: np.ndarray, masses, ppm: float) -> np.ndarray:
    parts = [cp.window_indices(pool_mass, float(m), ppm) for m in masses if np.isfinite(m)]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)


def build_libraries(cp, meta: pd.DataFrame, lib_mask: np.ndarray, pool: pd.DataFrame, val_mass, test_mass,
                    cfg: FusionConfig) -> Libraries:
    """Allocate centroids only for structures that can appear in a validation or test window."""
    pool_mass = pool.mass.to_numpy()
    in_train = (pool.formula.astype(str) != "").to_numpy()   # external candidates have no spectra
    val_need = window_union(cp, pool_mass, val_mass, cfg.ppm)
    val_need = val_need[pool.n_ref.to_numpy()[val_need] > 0]
    test_need = window_union(cp, pool_mass, test_mass, cfg.ppm)
    test_need = test_need[in_train[test_need]]
    key_index = pd.Series(np.arange(len(pool)), index=pool.key.to_numpy())
    row_pool = meta.inchikey14.map(key_index).fillna(-1).to_numpy().astype(np.int64)
    row_mode = mode_ids(meta.ionization_mode)
    usable = meta.good_label.to_numpy() & (meta.num_peaks.to_numpy() >= 3) & (row_pool >= 0)
    val = Centroids(val_need, len(pool), cfg.n_bins)
    test = Centroids(test_need, len(pool), cfg.n_bins)
    val_cid = np.where(usable & lib_mask, val.ids(row_pool, row_mode), -1)
    test_cid = np.where(usable, test.ids(row_pool, row_mode), -1)
    return Libraries(val, test, val_cid, test_cid)


def feature_pass(path: str, n_rows: int, libs: Libraries, wanted: dict, cfg: FusionConfig, log=print) -> dict:
    """One pass over train.parquet: fill both centroid libraries and return float16 matrices for ``wanted``.

    wanted: {name: row_ids}. Returns {name: (len(row_ids), n_bins) float16} in the order of row_ids.
    """
    select = (libs.val_cid >= 0) | (libs.test_cid >= 0)
    outs, pos = {}, {}
    for name, rows in wanted.items():
        rows = np.asarray(rows, dtype=np.int64)
        outs[name] = np.zeros((len(rows), cfg.n_bins), dtype=np.float16)
        where = np.full(n_rows, -1, dtype=np.int64)
        where[rows] = np.arange(len(rows))
        pos[name] = where
        select |= where >= 0
    log(f"feature pass: {int(select.sum()):,} of {n_rows:,} rows to bin")
    for k, (rows, X) in enumerate(iter_binned(path, select, cfg)):
        for cid, cent in ((libs.val_cid, libs.val), (libs.test_cid, libs.test)):
            c = cid[rows]
            keep = c >= 0
            cent.add(c[keep], X[keep])
        for name in wanted:
            p = pos[name][rows]
            keep = p >= 0
            outs[name][p[keep]] = X[keep]
        if k % 25 == 0:
            log(f"  binned up to row {int(rows[-1]):,}")
    libs.val.finish()
    libs.test.finish()
    return outs


# ---------- candidate scores ----------


@dataclass
class Scores:
    mols: list                       # molecule ids, in output order
    mass: np.ndarray                 # consensus neutral mass per molecule
    seg: np.ndarray                  # molecule number of each flat candidate row
    cand: np.ndarray                 # pool index of each flat candidate row
    src: dict = field(default_factory=dict)   # source -> flat raw scores
    conf: np.ndarray | None = None   # per molecule: best library score in its window
    truth: np.ndarray | None = None  # per molecule: flat row of the true structure, -1 if outside window

    @property
    def offsets(self) -> np.ndarray:
        return np.r_[0, np.cumsum(np.bincount(self.seg, minlength=len(self.mols)))]


def score_molecules(cp, spectra: pd.DataFrame, X: np.ndarray, probs: np.ndarray, cent: Centroids,
                    pool: pd.DataFrame, pool_fp: np.ndarray, cfg: FusionConfig, mols=None,
                    truth_keys: dict | None = None) -> Scores:
    """Score every candidate in each molecule's mass window with the three sources.

    spectra: one row per spectrum with molecule_id, ionization_mode, neutral_mass, positionally aligned
    with ``X`` (binned) and ``probs`` (predicted bit probabilities).
    """
    pool_mass = pool.mass.to_numpy()
    pool_keys = pool.key.to_numpy()
    modes = mode_ids(spectra.ionization_mode)
    nmass = spectra.neutral_mass.to_numpy(dtype=float)
    groups = spectra.reset_index(drop=True).groupby("molecule_id", sort=False).indices
    mols = list(pd.unique(spectra.molecule_id)) if mols is None else list(mols)
    segs, cands, conf, truth, masses = [], [], np.zeros(len(mols)), np.full(len(mols), -1), np.zeros(len(mols))
    src = {s: [] for s in SOURCES}
    flat = 0
    for i, mol in enumerate(mols):
        p = groups[mol]
        mass = float(np.nanmedian(nmass[p]))
        masses[i] = mass
        idx = cp.window_indices(pool_mass, mass, cfg.ppm) if np.isfinite(mass) else np.zeros(0, dtype=np.int64)
        if len(idx) > cfg.max_candidates:
            idx = np.sort(idx[np.argsort(np.abs(pool_mass[idx] - mass), kind="stable")[: cfg.max_candidates]])
        if not len(idx):
            continue
        lib = np.zeros(len(idx), dtype=np.float32)
        for m in np.unique(modes[p]):
            lib += cent.scores(idx, int(m), X[p[modes[p] == m]])
        lib /= len(p)
        pr = np.clip(probs[p].astype(np.float64).mean(axis=0), 1e-4, 1 - 1e-4)
        bits = np.unpackbits(pool_fp[idx], axis=1)[:, : len(pr)].astype(np.float32)
        overlap = bits @ pr.astype(np.float32)
        src["library"].append(lib)
        src["fp_ll"].append(bits @ np.log(pr / (1 - pr)).astype(np.float32))
        src["fp_tani"].append(overlap / (pr.sum() + bits.sum(axis=1) - overlap + 1e-8))
        segs.append(np.full(len(idx), i))
        cands.append(idx)
        conf[i] = float(lib.max())
        if truth_keys is not None:
            hit = np.flatnonzero(pool_keys[idx] == truth_keys[mol])
            truth[i] = flat + int(hit[0]) if len(hit) else -1
        flat += len(idx)

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    return Scores(mols=mols, mass=masses, seg=cat(segs, np.int64), cand=cat(cands, np.int64),
                  src={s: cat(v, np.float64) for s, v in src.items()}, conf=conf,
                  truth=truth if truth_keys is not None else None)


# ---------- normalisation, ranks, tuning ----------


def normalize(x: np.ndarray, seg: np.ndarray, n_mol: int, method: str, k: float = 60.0) -> np.ndarray:
    """Per-molecule z-score, or reciprocal rank 1/(k + rank) with ties sharing the best rank."""
    if method == "z":
        cnt = np.maximum(np.bincount(seg, minlength=n_mol), 1)
        mean = np.bincount(seg, x, n_mol) / cnt
        sd = np.sqrt(np.bincount(seg, (x - mean[seg]) ** 2, n_mol) / cnt)
        return np.where(sd[seg] > 1e-9, (x - mean[seg]) / np.maximum(sd[seg], 1e-9), 0.0)
    if method == "rrf":
        if not len(x):
            return x.astype(np.float64)
        order = np.lexsort((-x, seg))
        xs, ss = x[order], seg[order]
        n = np.arange(len(xs))
        new_seg = np.r_[True, ss[1:] != ss[:-1]]
        new_val = new_seg | np.r_[True, xs[1:] != xs[:-1]]
        rank = np.maximum.accumulate(np.where(new_val, n, 0)) - np.maximum.accumulate(np.where(new_seg, n, 0)) + 1
        out = np.empty(len(x), dtype=np.float64)
        out[order] = 1.0 / (k + rank)
        return out
    raise ValueError(method)


def reciprocal_ranks(f: np.ndarray, sc: Scores, top_k: int = 25) -> np.ndarray:
    """Per molecule 1/rank of the truth under scores ``f`` (ties broken by pool index, as in rankings)."""
    n = len(sc.mols)
    has = sc.truth >= 0
    tv = np.full(n, np.nan)
    tc = np.full(n, -1, dtype=np.int64)
    tv[has] = f[sc.truth[has]]
    tc[has] = sc.cand[sc.truth[has]]
    t = tv[sc.seg]
    ahead = (f > t) | ((f == t) & (sc.cand < tc[sc.seg]))
    rank = 1 + np.bincount(sc.seg, ahead.astype(np.float64), n)
    return np.where(has & (rank <= top_k), 1.0 / rank, 0.0)


def balance_weights(cls: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-molecule weights so that sum(w * rr) is the mean of the class 1 and class 2 MRR within ``mask``."""
    w = np.zeros(len(cls))
    present = [c for c in (1, 2) if (mask & (cls == c)).any()]
    for c in present:
        m = mask & (cls == c)
        w[m] = 1.0 / m.sum() / len(present)
    return w


def weight_grid(n_src: int, n_random: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fixed = [*np.eye(n_src), np.full(n_src, 1.0 / n_src)]
    return np.vstack([np.array(fixed), rng.dirichlet(np.ones(n_src), n_random)])


def _subset(N: np.ndarray, sc: Scores, keep_mol: np.ndarray):
    """Rows of ``N`` and a Scores view restricted to the molecules in ``keep_mol`` (faster tuning)."""
    rows = keep_mol[sc.seg]
    new_mol = np.cumsum(keep_mol) - 1
    new_row = np.cumsum(rows) - 1
    t = sc.truth[keep_mol]
    sub = Scores(mols=list(range(int(keep_mol.sum()))), mass=sc.mass[keep_mol], seg=new_mol[sc.seg[rows]],
                 cand=sc.cand[rows], truth=np.where(t >= 0, new_row[np.maximum(t, 0)], -1))
    return N[rows], sub


def _best(N: np.ndarray, sc: Scores, w_mol: np.ndarray, grid: np.ndarray):
    keep = w_mol > 0
    N, sc = _subset(N, sc, keep)
    w_mol = w_mol[keep]
    best, best_w = -1.0, grid[0]
    for w in grid:
        v = float((w_mol * reciprocal_ranks(N @ w, sc)).sum())
        if v > best + 1e-12:
            best, best_w = v, w
    return best_w, best


def conf_bins(conf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """0 = no library evidence in the window, then 1..len(edges)+1 by increasing confidence."""
    return np.where(conf > 0, 1 + np.digitize(conf, edges), 0)


def fit_variant(kind: str, N: np.ndarray, sc: Scores, cls: np.ndarray, mask: np.ndarray, cfg: FusionConfig) -> dict:
    grid = weight_grid(N.shape[1], cfg.n_random, cfg.seed)
    w_global, _ = _best(N, sc, balance_weights(cls, mask), grid)
    if kind == "global":
        return {"kind": kind, "weights": w_global}
    pos = sc.conf[mask & (sc.conf > 0)]
    qs = np.linspace(0, 1, cfg.n_conf_bins + 1)[1:-1]
    edges = np.unique(np.quantile(pos, qs)) if len(pos) else np.zeros(0)
    bins = conf_bins(sc.conf, edges)
    W = np.tile(w_global, (len(edges) + 2, 1))
    sizes = []
    for b in range(len(edges) + 2):
        m = mask & (bins == b)
        sizes.append(int(m.sum()))
        if m.sum() >= cfg.min_bin:
            # weight each molecule as in the balanced objective over the whole mask
            W[b], _ = _best(N, sc, balance_weights(cls, mask) * m, grid)
    return {"kind": kind, "weights": w_global, "edges": edges, "bin_weights": W, "bin_sizes": sizes}


def apply_variant(params: dict, N: np.ndarray, sc: Scores) -> np.ndarray:
    if params["kind"] == "global":
        return N @ params["weights"]
    W = params["bin_weights"][conf_bins(sc.conf, params["edges"])]
    return (N * W[sc.seg]).sum(axis=1)


def folds(mols) -> np.ndarray:
    return np.array([zlib.crc32(str(m).encode()) % 2 for m in mols])


def cross_fit(kind: str, N: np.ndarray, sc: Scores, cls: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    """Scores where each molecule uses parameters tuned on the other fold."""
    fold = folds(sc.mols)
    out = np.zeros(len(sc.seg))
    for k in (0, 1):
        params = fit_variant(kind, N, sc, cls, fold == k, cfg)
        rows = fold[sc.seg] != k
        out[rows] = apply_variant(params, N, sc)[rows]
    return out


def source_matrix(sc: Scores, method: str, cfg: FusionConfig) -> np.ndarray:
    return np.stack([normalize(sc.src[s], sc.seg, len(sc.mols), method, cfg.rrf_k) for s in SOURCES], axis=1)


def rankings(cp, sc: Scores, f: np.ndarray, pool: pd.DataFrame) -> dict:
    """molecule_id -> pool indices, best first, filled to 25 with mass neighbours (Member 3's ``fill_to_k``)."""
    pool_mass = pool.mass.to_numpy()
    off = sc.offsets
    out = {}
    for i, mol in enumerate(sc.mols):
        idx, fi = sc.cand[off[i] : off[i + 1]], f[off[i] : off[i + 1]]
        ranked = idx[np.lexsort((idx, -fi))][: cp.TOP_K] if len(idx) else []
        out[mol] = cp.fill_to_k(ranked, pool_mass, sc.mass[i]) if np.isfinite(sc.mass[i]) else list(ranked)
    return out


# ---------- validation report and test application ----------


def _row(name: str, table: pd.DataFrame, **extra) -> dict:
    b = table.loc["balanced"]
    row = {"method": name, "balanced": b["MRR@25"], "class 1": table.loc["class 1", "MRR@25"],
           "class 2": table.loc["class 2", "MRR@25"], "np proxy": table.loc["np proxy", "MRR@25"],
           "top1": b["top1"], "top25": b["top25"]}
    if "MRR_ci_low" in table.columns:
        row["CI"] = f"{b['MRR_ci_low']:.3f}-{b['MRR_ci_high']:.3f}"
    row.update(extra)
    return row


def tune_and_evaluate(cp, sc: Scores, truth: pd.DataFrame, pool: pd.DataFrame, cfg: FusionConfig, log=print) -> dict:
    """Evaluate each source and each fusion variant with ``cp.evaluate``; pick the best by honest score.

    Single sources need no tuning. For fusion variants, ``balanced`` is the cross-fitted score and
    ``in_sample`` the score with parameters fitted on all validation molecules.
    """
    truth = truth.set_index("molecule_id").loc[sc.mols].reset_index()
    cls = truth.cls.to_numpy()
    keys = pool.key.to_numpy()
    rows, params, tables, val_scores = [], {}, {}, {}
    for s in SOURCES:
        t = cp.evaluate(rankings(cp, sc, sc.src[s], pool), truth, keys, ci=True)
        tables[s] = t
        rows.append(_row(s, t, kind="single"))
        params[s] = {"kind": "single", "source": s}
    for method in METHODS:
        N = source_matrix(sc, method, cfg)
        for kind in KINDS:
            name = f"{kind}_{method}"
            full = fit_variant(kind, N, sc, cls, np.ones(len(cls), dtype=bool), cfg)
            f_cf = cross_fit(kind, N, sc, cls, cfg)
            t_cf = cp.evaluate(rankings(cp, sc, f_cf, pool), truth, keys, ci=True)
            t_in = cp.evaluate(rankings(cp, sc, apply_variant(full, N, sc), pool), truth, keys)
            tables[name] = t_cf
            val_scores[name] = f_cf
            params[name] = {**full, "method": method}
            rows.append(_row(name, t_cf, kind="fusion (cross-fitted)", in_sample=t_in.loc["balanced", "MRR@25"]))
            log(f"{name}: cross-fitted {t_cf.loc['balanced', 'MRR@25']:.4f} | in-sample "
                f"{t_in.loc['balanced', 'MRR@25']:.4f} | weights {np.round(full['weights'], 3).tolist()}")
    summary = pd.DataFrame(rows).sort_values("balanced", ascending=False).reset_index(drop=True)
    best = summary.method[0]
    return {"summary": summary, "best": best, "params": params, "tables": tables, "val_scores": val_scores}


def add_row(report: dict, name: str, table: pd.DataFrame, kind: str) -> None:
    """Add a reference row (e.g. B1) to the summary without making it selectable."""
    s = pd.concat([report["summary"], pd.DataFrame([_row(name, table, kind=kind)])], ignore_index=True)
    report["summary"] = s.sort_values("balanced", ascending=False).reset_index(drop=True)
    report["tables"][name] = table


def apply_best(report: dict, sc: Scores, cfg: FusionConfig, name: str | None = None) -> np.ndarray:
    """Fused scores for new molecules (e.g. the test set) with the chosen variant's full-validation fit."""
    p = report["params"][name or report["best"]]
    if p["kind"] == "single":
        return sc.src[p["source"]]
    return apply_variant(p, source_matrix(sc, p["method"], cfg), sc)


def coverage(sc: Scores, truth: pd.DataFrame | None = None) -> dict:
    """Window sizes and, for validation, the share of molecules whose truth is in the window."""
    sizes = np.bincount(sc.seg, minlength=len(sc.mols))
    out = {"molecules": len(sc.mols), "candidates_median": float(np.median(sizes)) if len(sizes) else 0.0,
           "candidates_max": int(sizes.max()) if len(sizes) else 0, "no_candidates": int((sizes == 0).sum()),
           "with_library_evidence": round(float((sc.conf > 0).mean()), 4) if len(sc.mols) else 0.0}
    if sc.truth is not None and truth is not None:
        cls = truth.set_index("molecule_id").loc[sc.mols].cls.to_numpy()
        hit = sc.truth >= 0
        out["truth_in_window"] = round(float(hit.mean()), 4)
        for c in (1, 2):
            if (cls == c).any():
                out[f"truth_in_window_class{c}"] = round(float(hit[cls == c].mean()), 4)
    return out


def score_table(sc: Scores, pool: pd.DataFrame, fused: np.ndarray) -> pd.DataFrame:
    """Long table in the team format (query_id, inchikey14, score) plus each source's raw score."""
    df = pd.DataFrame({"query_id": np.asarray(sc.mols, dtype=object)[sc.seg],
                       "inchikey14": pool.key.to_numpy()[sc.cand], "score": fused})
    for s in SOURCES:
        df[s] = sc.src[s]
    return df


def describe(params: dict) -> dict:
    """JSON-friendly copy of a variant's parameters."""
    out = {}
    for k, v in params.items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    if "weights" in out:
        out["weights"] = dict(zip(SOURCES, np.round(params["weights"], 4).tolist(), strict=True))
    return out


__all__ = [
    "FusionConfig", "Libraries", "Centroids", "Scores", "SOURCES", "side_features", "iter_binned", "bin_file",
    "build_libraries", "feature_pass", "score_molecules", "normalize", "reciprocal_ranks", "tune_and_evaluate",
    "add_row", "apply_best", "rankings", "coverage", "score_table", "describe",
]
