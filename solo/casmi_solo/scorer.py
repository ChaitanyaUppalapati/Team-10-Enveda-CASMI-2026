"""MRR@25 exactly as the competition defines it, plus recall@K and per-class tables.

Rankings are lists of official keys (``casmi_solo.chem.official_key``), best first. Duplicate keys
are removed (keeping the first) before truncating to 25, as the metric gives no credit for repeats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_K = 25


def dedup(keys) -> list:
    seen, out = set(), []
    for k in keys:
        if k is not None and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def rank_of_truth(ranked_keys, truth_key) -> int:
    """1-based rank of the truth in the deduplicated list, 0 if absent."""
    for i, k in enumerate(dedup(ranked_keys), 1):
        if k == truth_key:
            return i
    return 0


def evaluate(rankings: dict, truth: pd.DataFrame, ks=(1, 5, 25, 100)) -> pd.DataFrame:
    """``truth`` has one row per molecule: molecule_id, key (official), cls. Returns a table by class."""
    r = np.array([rank_of_truth(rankings.get(m, []), k) for m, k in zip(truth.molecule_id, truth.key, strict=True)])
    df = truth[["molecule_id", "cls"]].assign(rank=r)
    df["rr"] = np.where((df["rank"] > 0) & (df["rank"] <= TOP_K), 1.0 / np.maximum(df["rank"], 1), 0.0)
    for k in ks:
        df[f"hit@{k}"] = (df["rank"] > 0) & (df["rank"] <= k)

    def table(g):
        row = {"n": len(g), "MRR@25": g.rr.mean()}
        row.update({f"R@{k}": g[f"hit@{k}"].mean() for k in ks})
        return pd.Series(row)

    out = df.groupby("cls").apply(table, include_groups=False)
    out.loc["balanced"] = out.mean()
    out.loc["balanced", "n"] = len(df)
    return out.round(4), df


def write_submission(rankings_smiles: dict, molecule_ids, path) -> pd.DataFrame:
    """rankings_smiles: molecule_id -> list of SMILES (already deduplicated by official key)."""
    rows = [{"molecule_id": m, "smiles": ";".join(rankings_smiles.get(m, [])[:TOP_K]) or "C"} for m in molecule_ids]
    sub = pd.DataFrame(rows)
    check_submission(sub, molecule_ids)
    sub.to_csv(path, index=False)
    return sub


def check_submission(sub: pd.DataFrame, molecule_ids) -> None:
    assert list(sub.columns) == ["molecule_id", "smiles"], sub.columns
    assert sub.molecule_id.is_unique and set(sub.molecule_id) == set(molecule_ids)
    assert sub.smiles.notna().all() and (sub.smiles.str.len() > 0).all()
    assert (sub.smiles.str.count(";") < TOP_K).all()
