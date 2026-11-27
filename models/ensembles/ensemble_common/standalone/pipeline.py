"""Self-contained late-fusion pipeline (see package docstring).

Validation is structure-level and reports two sets, mirroring the proposal:
  * ``absent``  : held-out structures. No spectrum of them is in the library or the fingerprint
                  training set (reference-absent).
  * ``covered`` : training structures whose query spectra are removed from the library
                  (leave-query-out) and from fingerprint training (reference-covered).
Fusion weights are tuned on both sets together. Candidates = structures seen in training,
filtered by precursor mass; true structures outside that pool are impossible to find (coverage).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from casmi.data import iter_spectra
from casmi.interfaces import INCHIKEY14, QUERY_ID, SCORE, TOP_K, rank_candidates

from ..fusion import fuse, tune_weights
from ..inputs import align_scores
from ..metrics import mrr_at_k, reciprocal_ranks
from ..normalize import normalize
from .chem import exact_mass, morgan_bits, neutral_mass
from .library import CentroidLibrary
from .spectra import bin_one, pool

SOURCES = ("library", "fingerprint")


@dataclass
class Config:
    train_path: str | None = None
    test_path: str | None = None
    max_mz: float = 1500.0
    bin_width: float = 1.0
    n_bits: int = 2048
    ppm: float = 20.0
    min_tol_da: float = 0.003
    max_candidates: int = 2000
    keep_top: int = 200            # per query and source, before fusion
    heldout_permille: int = 100    # 10% of structures held out (reference-absent)
    covered_permille: int = 30     # ~3% of structures reserved for covered validation
    max_absent_queries: int = 1500
    max_covered_queries: int = 1500
    query_spectra: int = 3         # spectra pooled per validation query
    fp_per_structure: int = 2
    max_fp_train: int = 200_000
    max_batches: int | None = None  # debugging: stop the training pass early
    batch_size: int = 20_000
    seed: int = 2026
    method: str = "rrf"
    n_random: int = 200


@dataclass
class State:
    structs: pd.DataFrame                  # inchikey14, smiles, mass, heldout
    fps: np.ndarray                        # (n_struct, n_bits) uint8
    library: CentroidLibrary
    absent_q: list = field(default_factory=list)    # dicts: query_id, truth_idx, vecs, prec, mass
    covered_q: list = field(default_factory=list)   # + sub_sum, sub_n
    fp_X: np.ndarray | None = None
    fp_Y: np.ndarray | None = None
    index: dict = field(default_factory=dict)
    order: np.ndarray | None = None        # structure indices sorted by mass
    sorted_mass: np.ndarray | None = None


def _role(key: str, cfg: Config) -> str:
    h = zlib.crc32(f"{cfg.seed}:{key}".encode()) % 1000
    if h < cfg.heldout_permille:
        return "heldout"
    if h < cfg.heldout_permille + cfg.covered_permille:
        return "covered"
    return "train"


def _rows(batch: pd.DataFrame):
    for mz, it, prec, ad, key in zip(batch.mzs, batch.intensities, batch.precursor_mz, batch.adduct,
                                     batch[INCHIKEY14], strict=True):
        yield mz, it, float(prec), ad, key


def _smiles_col(batch: pd.DataFrame) -> pd.Series:
    return batch["smiles"] if "smiles" in batch else batch["normalized_smiles"]


def scan_structures(cfg: Config) -> pd.DataFrame:
    """Pass 0: unique training structures with SMILES, RDKit mass and role."""
    seen: dict[str, str] = {}
    for b in iter_spectra("train", ["inchikey14", "smiles"], path=cfg.train_path, batch_size=200_000):
        for key, smi in zip(b[INCHIKEY14], _smiles_col(b), strict=True):
            if isinstance(key, str) and isinstance(smi, str) and key not in seen:
                seen[key] = smi
    df = pd.DataFrame({INCHIKEY14: list(seen), "smiles": list(seen.values())})
    df["mass"] = [exact_mass(s) for s in df.smiles]
    df = df[np.isfinite(df["mass"])].reset_index(drop=True)
    df["role"] = [_role(k, cfg) for k in df[INCHIKEY14]]
    df["heldout"] = df["role"] == "heldout"
    return df


def build_state(cfg: Config, log=print) -> State:
    """Pass 0 + pass 1: structure table, fingerprints, library sums, validation queries, fp training set."""
    structs = scan_structures(cfg)
    index = {k: i for i, k in enumerate(structs[INCHIKEY14])}
    log(f"{len(structs)} structures ({int(structs.heldout.sum())} held out)")
    fps = np.stack([morgan_bits(s, n_bits=cfg.n_bits) for s in structs.smiles])
    dim = int(np.ceil(cfg.max_mz / cfg.bin_width))
    order = np.argsort(structs["mass"].to_numpy(), kind="stable")
    st = State(structs=structs, fps=fps, library=CentroidLibrary(len(structs), dim), index=index,
               order=order, sorted_mass=structs["mass"].to_numpy()[order])

    role = structs["role"].to_numpy()
    n_seen = np.zeros(len(structs), dtype=np.int32)
    abs_buf: dict[int, dict] = {}
    cov_buf: dict[int, dict] = {}
    fp_X, fp_Y, fp_n = [], [], np.zeros(len(structs), dtype=np.int8)
    cols = ["mzs", "intensities", "precursor_mz", "adduct", "inchikey14"]
    for nb, batch in enumerate(iter_spectra("train", cols, path=cfg.train_path, batch_size=cfg.batch_size)):
        if cfg.max_batches is not None and nb >= cfg.max_batches:
            break
        for mz, it, prec, ad, key in _rows(batch):
            i = index.get(key)
            if i is None:
                continue
            vec = bin_one(mz, it, cfg.max_mz, cfg.bin_width)
            if not vec.any():
                continue
            m = neutral_mass(prec, ad)
            r = role[i]
            if r == "heldout":
                if i in abs_buf or len(abs_buf) < cfg.max_absent_queries:
                    q = abs_buf.setdefault(i, {"vecs": [], "prec": [], "mass": []})
                    if len(q["vecs"]) < cfg.query_spectra:
                        q["vecs"].append(vec), q["prec"].append(prec), q["mass"].append(m)
                continue
            st.library.add(i, vec)
            n_seen[i] += 1
            if r == "covered" and (i in cov_buf or len(cov_buf) < cfg.max_covered_queries):
                q = cov_buf.setdefault(i, {"vecs": [], "prec": [], "mass": []})
                if len(q["vecs"]) < cfg.query_spectra:
                    q["vecs"].append(vec), q["prec"].append(prec), q["mass"].append(m)
                    continue  # reserved for validation: not used to train the fingerprint net
            if r != "covered" and fp_n[i] < cfg.fp_per_structure and len(fp_X) < cfg.max_fp_train:
                fp_X.append(np.append(vec, prec / cfg.max_mz).astype(np.float16))
                fp_Y.append(i)
                fp_n[i] += 1
        log(f"batch {nb + 1}: library spectra={int(n_seen.sum())} fp_train={len(fp_X)}")
    st.fp_X = np.stack(fp_X) if fp_X else np.zeros((0, dim + 1), dtype=np.float16)
    st.fp_Y = fps[np.array(fp_Y, dtype=np.int64)] if fp_Y else np.zeros((0, cfg.n_bits), dtype=np.uint8)
    for i, q in abs_buf.items():
        st.absent_q.append({"query_id": f"abs:{structs[INCHIKEY14][i]}", "truth": i, **_pack(q)})
    for i, q in cov_buf.items():
        # need at least one other library spectrum after removing the query's own
        if st.library.counts[i] > len(q["vecs"]):
            sub = np.sum(q["vecs"], axis=0).astype(np.float32)
            st.covered_q.append({"query_id": f"cov:{structs[INCHIKEY14][i]}", "truth": i,
                                 "sub": (i, sub, len(q["vecs"])), **_pack(q)})
    log(f"validation queries: absent={len(st.absent_q)} covered={len(st.covered_q)}")
    return st


def _pack(q: dict) -> dict:
    return {"vecs": np.stack(q["vecs"]), "prec": np.array(q["prec"]), "mass": float(np.median(q["mass"]))}


def candidate_idx(mass: float, st: State, cfg: Config) -> np.ndarray:
    tol = max(cfg.ppm * 1e-6 * mass, cfg.min_tol_da)
    lo, hi = np.searchsorted(st.sorted_mass, [mass - tol, mass + tol])
    cand = st.order[lo:hi]
    if len(cand) > cfg.max_candidates:
        diff = np.abs(st.structs["mass"].to_numpy()[cand] - mass)
        cand = cand[np.argsort(diff, kind="stable")[: cfg.max_candidates]]
    return cand


def _top(idx: np.ndarray, score: np.ndarray, n: int):
    if len(idx) > n:
        keep = np.argpartition(-score, n - 1)[:n]
        return idx[keep], score[keep]
    return idx, score


def score_queries(queries: list[dict], st: State, cfg: Config, fp_model, *, library_absent: bool,
                  log=print) -> dict[str, pd.DataFrame]:
    """CandidateScores per source for ``queries`` (library cosine; fingerprint log-likelihood)."""
    keys = st.structs[INCHIKEY14].to_numpy()
    absent = st.structs["heldout"].to_numpy() if library_absent else None
    rows = {s: [] for s in SOURCES}
    probs = None
    if queries:
        flat = np.concatenate([np.hstack([q["vecs"], (q["prec"] / cfg.max_mz)[:, None]]) for q in queries])
        probs = fp_model.predict(flat)
    offset = 0
    for q in queries:
        n = len(q["vecs"])
        p = np.clip(probs[offset : offset + n].mean(axis=0), 1e-4, 1 - 1e-4)
        offset += n
        cand = candidate_idx(q["mass"], st, cfg)
        if not len(cand):
            continue
        lib = st.library.scores(pool(q["vecs"]), cand, absent=absent, subtract=q.get("sub"))
        w = np.log(p) - np.log1p(-p)
        fpv = st.fps[cand].astype(np.float32) @ w.astype(np.float32)
        for name, sc in (("library", lib), ("fingerprint", fpv)):
            ci, cs = _top(cand, sc, cfg.keep_top)
            rows[name].append(pd.DataFrame({QUERY_ID: q["query_id"], INCHIKEY14: keys[ci], SCORE: cs.astype(float)}))
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=[QUERY_ID, INCHIKEY14, SCORE]))
            for k, v in rows.items()}


def truth_frame(queries: list[dict], st: State) -> pd.DataFrame:
    keys = st.structs[INCHIKEY14].to_numpy()
    return pd.DataFrame({QUERY_ID: [q["query_id"] for q in queries], INCHIKEY14: [keys[q["truth"]] for q in queries]})


def coverage(scores_by_source: dict[str, pd.DataFrame], truth: pd.DataFrame) -> float:
    """Share of queries whose true structure is in any source's candidate list."""
    both = pd.concat(scores_by_source.values())[[QUERY_ID, INCHIKEY14]]
    hit = truth.merge(both.drop_duplicates(), on=[QUERY_ID, INCHIKEY14])
    return float(hit[QUERY_ID].nunique() / max(len(truth), 1))


def tune_and_report(st: State, cfg: Config, fp_model, log=print) -> dict:
    """Score both validation sets, tune fusion weights on them together, return a report dict."""
    sets = {"absent": (st.absent_q, True), "covered": (st.covered_q, False)}
    parts, truths, report = [], [], {"per_source": {}, "coverage": {}}
    for name, (qs, lib_absent) in sets.items():
        sc = score_queries(qs, st, cfg, fp_model, library_absent=lib_absent, log=log)
        tr = truth_frame(qs, st)
        report["coverage"][name] = coverage(sc, tr)
        report["per_source"][name] = {s: mrr_at_k(sc[s], tr) for s in SOURCES}
        parts.append(align_scores(sc)), truths.append(tr)
        log(f"{name}: n={len(qs)} coverage={report['coverage'][name]:.3f} {report['per_source'][name]}")
    wide, truth = pd.concat(parts, ignore_index=True), pd.concat(truths, ignore_index=True)
    norm = normalize(wide, list(SOURCES), cfg.method)
    weights, best = tune_weights(norm, truth, list(SOURCES), n_random=cfg.n_random, seed=cfg.seed)
    fused = fuse(norm, weights)
    report.update(weights=weights, method=cfg.method, fused_mrr_both=best)
    for name, tr in zip(sets, truths, strict=True):
        sub = fused[fused[QUERY_ID].isin(tr[QUERY_ID])]
        report.setdefault("fused", {})[name] = mrr_at_k(sub, tr)
    report["fused_scores"] = fused
    return report


def predict_test(st: State, cfg: Config, fp_model, weights: dict, log=print) -> pd.DataFrame:
    """Fuse the two sources for the competition test spectra; returns top-25 CandidateScores."""
    by_q: dict[str, dict] = {}
    cols = ["mzs", "intensities", "precursor_mz", "adduct", "query_id"]
    for b in iter_spectra("test", cols, path=cfg.test_path, batch_size=cfg.batch_size):
        for mz, it, prec, ad, qid in zip(b.mzs, b.intensities, b.precursor_mz, b.adduct, b.query_id, strict=True):
            q = by_q.setdefault(qid, {"query_id": qid, "vecs": [], "prec": [], "mass": []})
            q["vecs"].append(bin_one(mz, it, cfg.max_mz, cfg.bin_width))
            q["prec"].append(float(prec)), q["mass"].append(neutral_mass(prec, ad))
    queries = [{**_pack(q), "query_id": qid} for qid, q in by_q.items()]
    log(f"test queries: {len(queries)}")
    sc = score_queries(queries, st, cfg, fp_model, library_absent=False, log=log)
    norm = normalize(align_scores(sc), list(SOURCES), cfg.method)
    fused = fuse(norm, weights)
    return rank_candidates(fused, top_k=TOP_K)[[QUERY_ID, INCHIKEY14, SCORE]]


def attach_smiles(scores: pd.DataFrame, st: State) -> pd.DataFrame:
    """Add each candidate's SMILES (first structure per inchikey14) for the submission table."""
    lookup = st.structs.drop_duplicates(INCHIKEY14).set_index(INCHIKEY14)["smiles"]
    out = rank_candidates(scores, top_k=TOP_K)
    out["smiles"] = out[INCHIKEY14].map(lookup)
    return out


__all__ = ["Config", "State", "build_state", "tune_and_report", "predict_test", "attach_smiles", "reciprocal_ranks"]
