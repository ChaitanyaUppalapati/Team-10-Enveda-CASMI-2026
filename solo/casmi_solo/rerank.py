"""Candidate features per (molecule, candidate) and the LightGBM LambdaRank reranker.

Features combine every scorer we have, so the reranker learns e.g. when to trust the library (B2)
and when to fall back on the fingerprint model (S3) instead of a hand-tuned gate:

- mass: |ppm| error, number of candidates in the window
- priors: training spectra count, source membership, NP-likeness, literature counts
- B2: best library cosine (-1 = no spectra), its rank in the molecule, the molecule's best B2 score
- S3: fingerprint log-likelihood and cosine, their gaps to the molecule's best, their ranks
- query: number of spectra, ionisation mode
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import PPM, b2_scores
from .fingerprints import unpack
from .fpmodel import fp_cosine_scores, fp_loglik_scores

PRIOR_COLS = ["n_spectra", "in_train", "in_coconut", "in_pcl", "np_likeness", "coconut_annot", "pubmed", "patent",
              "annot_types", "in_pc", "pc_pmid", "prior"]


def _rank_desc(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-x, kind="stable"), kind="stable") + 1


def molecule_features(mol, mass, pool, lib, Q, modes, allowed, logits, fp, fp_index, ppm=PPM) -> pd.DataFrame:
    """All window candidates of one molecule with features. ``Q``/``modes``/``logits`` are its spectra."""
    w = pool.window(mass, ppm)
    if len(w) == 0:
        return pd.DataFrame()
    t = pool.table.iloc[w]
    f = pd.DataFrame({"molecule_id": mol, "key": t.key.to_numpy()})
    f["ppm_abs"] = np.abs(t.mass.to_numpy() - mass) / mass * 1e6
    f["n_window"] = len(w)
    for c in PRIOR_COLS:
        f[c] = t[c].to_numpy() if c in t else 0.0
    f["n_spectra"] = np.log1p(f.n_spectra)
    f["pubmed"], f["patent"], f["pc_pmid"] = np.log1p(f.pubmed), np.log1p(f.patent), np.log1p(f.pc_pmid)

    b2 = b2_scores(pool, lib, w, Q, modes, allowed)
    f["b2"] = b2
    f["b2_rank"] = _rank_desc(b2)
    f["b2_mol_max"] = b2.max()
    f["b2_gap"] = b2.max() - b2

    if logits is not None:
        bits = unpack(fp[[fp_index[k] for k in t.key]]).astype(np.float32)
        lm = logits.mean(0)
        ll = fp_loglik_scores(lm, bits)
        cs = fp_cosine_scores(lm, bits)
        f["s3_ll"], f["s3_cos"] = ll, cs
        f["s3_ll_gap"], f["s3_cos_gap"] = ll.max() - ll, cs.max() - cs
        f["s3_ll_rank"], f["s3_cos_rank"] = _rank_desc(ll), _rank_desc(cs)
    f["n_query_spectra"] = Q.shape[0]
    f["mode_pos"] = float(np.mean(modes))
    return f


FEATURES = ["ppm_abs", "n_window", *PRIOR_COLS, "b2", "b2_rank", "b2_mol_max", "b2_gap", "s3_ll", "s3_cos",
            "s3_ll_gap", "s3_cos_gap", "s3_ll_rank", "s3_cos_rank", "n_query_spectra", "mode_pos"]


def train_ranker(train: pd.DataFrame, features=FEATURES, seed=2026, rounds=600):
    import lightgbm as lgb

    train = train.sort_values("molecule_id", kind="stable")
    groups = train.groupby("molecule_id", sort=False).size().to_numpy()
    ds = lgb.Dataset(train[features], label=train.label.astype(int), group=groups)
    params = {"objective": "lambdarank", "metric": "map", "eval_at": [25], "learning_rate": 0.05,
              "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambdarank_truncation_level": 30, "seed": seed, "verbose": -1}
    return lgb.train(params, ds, num_boost_round=rounds)


def rankings_from_scores(feats: pd.DataFrame, score: np.ndarray) -> dict:
    d = feats[["molecule_id", "key"]].assign(s=score).sort_values(["molecule_id", "s"], ascending=[True, False])
    return d.groupby("molecule_id", sort=False).key.apply(list).to_dict()
