"""Literature/patent 'popularity' priors from PubChemLite counts. Owner: Member 4.

PubChemLite carries ``PubMed_Count``, ``Patent_Count`` and ``AnnoTypeCount`` per compound. Well-documented compounds are
more likely to be the true answer among same-mass candidates, so ``log1p`` of these counts is offered as a small
re-ranking prior. Structures outside PubChemLite get 0 (undocumented as far as this table knows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODES = ("member", "pubmed", "patent", "both", "annotations")
COLUMNS = ["FirstBlock", "PubMed_Count", "Patent_Count", "AnnoTypeCount"]


def prior_values(table: pd.DataFrame, mode: str) -> pd.Series:
    """Prior per ``FirstBlock`` (= inchikey14) for ``mode``: member, pubmed, patent, both or annotations."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    pubmed = pd.to_numeric(table["PubMed_Count"], errors="coerce").fillna(0).clip(lower=0)
    patent = pd.to_numeric(table["Patent_Count"], errors="coerce").fillna(0).clip(lower=0)
    if mode == "member":
        values = np.ones(len(table))
    elif mode == "pubmed":
        values = np.log1p(pubmed)
    elif mode == "patent":
        values = np.log1p(patent)
    elif mode == "both":
        values = np.log1p(pubmed + patent)
    else:
        values = pd.to_numeric(table["AnnoTypeCount"], errors="coerce").fillna(0)
    return pd.Series(np.asarray(values, dtype=np.float32), index=table["FirstBlock"].to_numpy())


def pool_prior(pool_keys, table: pd.DataFrame, mode: str) -> np.ndarray:
    """Prior aligned with ``pool_keys`` (0 for structures not in the table)."""
    values = prior_values(table, mode)
    values = values[~values.index.duplicated()]
    return values.reindex(pd.Index(pool_keys)).fillna(0.0).to_numpy(dtype=np.float32)


def load_table(path) -> pd.DataFrame:
    """Counts table from the PubChemLite CSV or from the slim parquet written by :func:`save_counts`."""
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=COLUMNS)
    return pd.read_csv(path, usecols=COLUMNS, dtype={"FirstBlock": str})


def save_counts(csv_path, out_path) -> pd.DataFrame:
    """Write the four count columns as a small parquet (about 10 MB) for the Kaggle dataset."""
    table = load_table(csv_path)
    for col in COLUMNS[1:]:
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0).astype("int64")
    table.to_parquet(out_path, index=False)
    return table


def config_prior(pool_keys, prior_cfg: dict | None, candidates_dir) -> tuple[np.ndarray | None, dict]:
    """Prior array and ranking kwargs from ``predict.prior`` (``mode``, ``weight``, ``topk``, ``table``).

    Returns ``(None, {})`` when the prior is off (missing config or weight 0).
    """
    if not prior_cfg or not prior_cfg.get("weight"):
        return None, {}
    from pathlib import Path

    path = Path(candidates_dir) / prior_cfg.get("table", "pubchemlite_counts.parquet")
    if not path.is_file():
        raise FileNotFoundError(f"prior table {path} not found; build it with priors.save_counts")
    prior = pool_prior(pool_keys, load_table(path), prior_cfg.get("mode", "pubmed"))
    return prior, {"prior_weight": float(prior_cfg["weight"]), "prior_topk": prior_cfg.get("topk")}
