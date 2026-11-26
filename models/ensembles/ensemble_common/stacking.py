"""Stacked meta-learner over many models' scores (candidate-level learning to rank)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from casmi.interfaces import INCHIKEY14, QUERY_ID, SCORE

from .normalize import normalize


def build_features(wide: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    """Feature table: per model rrf, zscore, a 'scored' flag, plus the candidate count per query."""
    rrf = normalize(wide, model_ids, "rrf")
    z = normalize(wide, model_ids, "zscore")
    feats = wide[[QUERY_ID, INCHIKEY14]].copy()
    for m in model_ids:
        feats[f"{m}__rrf"] = rrf[m].to_numpy()
        feats[f"{m}__z"] = z[m].to_numpy()
        feats[f"{m}__seen"] = wide[m].notna().astype(float).to_numpy()
    feats["n_cand"] = wide.groupby(QUERY_ID)[INCHIKEY14].transform("size").astype(float).to_numpy()
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in (QUERY_ID, INCHIKEY14)]


def add_labels(feats: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    key = truth[[QUERY_ID, INCHIKEY14]].assign(_y=1.0)
    y = feats[[QUERY_ID, INCHIKEY14]].merge(key, on=[QUERY_ID, INCHIKEY14], how="left")["_y"]
    return y.fillna(0.0).astype(int)


def make_model(kind: str, seed: int):
    """``lgbm_rank`` (LambdaRank) or ``logreg`` (a safe, dependency-light fallback)."""
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, C=1.0)
    if kind == "lgbm_rank":
        import lightgbm as lgb

        return lgb.LGBMRanker(
            objective="lambdarank", n_estimators=200, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, random_state=seed, verbose=-1,
        )
    raise ValueError(f"unknown meta-learner {kind!r}")


def _fit(model, kind: str, X: pd.DataFrame, y: pd.Series, qids: pd.Series):
    """Fit on queries that have at least one positive (others carry no ranking signal)."""
    has_pos = y.groupby(qids).transform("max") > 0
    X, y, qids = X[has_pos], y[has_pos], qids[has_pos]
    if kind == "logreg":
        model.fit(X, y)
    else:
        order = qids.to_numpy().argsort(kind="stable")
        X, y, qids = X.iloc[order], y.iloc[order], qids.iloc[order]
        group = qids.groupby(qids, sort=False).size().to_numpy()
        model.fit(X, y, group=group)
    return model


def _predict(model, kind: str, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if kind == "logreg" else model.predict(X)


def out_of_fold_scores(
    feats: pd.DataFrame, y: pd.Series, kind: str = "lgbm_rank", n_splits: int = 5, seed: int = 2026
) -> pd.DataFrame:
    """Score every row with a model that never saw its query (folds split by query_id)."""
    cols = feature_columns(feats)
    qids = feats[QUERY_ID]
    uniq = qids.drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % n_splits, strict=True))
    fold = qids.map(fold_of).to_numpy()
    pred = np.zeros(len(feats))
    for f in range(n_splits):
        tr, te = fold != f, fold == f
        if not te.any() or not tr.any():
            continue
        model = _fit(make_model(kind, seed), kind, feats.loc[tr, cols], y[tr], qids[tr])
        pred[te] = _predict(model, kind, feats.loc[te, cols])
    return pd.DataFrame({QUERY_ID: qids, INCHIKEY14: feats[INCHIKEY14], SCORE: pred})


def fit_final(feats: pd.DataFrame, y: pd.Series, kind: str = "lgbm_rank", seed: int = 2026):
    cols = feature_columns(feats)
    return _fit(make_model(kind, seed), kind, feats[cols], y, feats[QUERY_ID]), cols


def score_with(model, cols: list[str], feats: pd.DataFrame, kind: str = "lgbm_rank") -> pd.DataFrame:
    return pd.DataFrame(
        {QUERY_ID: feats[QUERY_ID], INCHIKEY14: feats[INCHIKEY14], SCORE: _predict(model, kind, feats[cols])}
    )
