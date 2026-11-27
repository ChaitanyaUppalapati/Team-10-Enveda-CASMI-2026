"""Frozen validation split that imitates the hidden test set.

The hidden test is ~400 natural-product-like molecules measured on a Bruker timsTOF, 1-16 spectra each
(median 3), in three hidden novelty classes. We build two query groups:

- ``c1`` (class 1, "structure has public spectra"): the 250 structures of ``enveda-np-examples``
  (timsTOF, natural products, same pipeline as the test). Their np-examples spectra become the queries
  and are removed from the search library and from training; their spectra in *other* libraries stay,
  exactly as a class-1 test molecule has public spectra somewhere else.
- ``c2`` (class 2, "known structure, never measured"): a random 10% of the remaining structures is
  held out completely. None of their spectra may be used for training or library search, and they
  are only findable through an external candidate list. Queries are drawn from them, half from the
  timsTOF library (``enveda-180``) and half from natural-product libraries on other instruments.

Each query molecule is a random subset of one structure's spectra from one library, with a test-like
set size (geometric, median 3, capped at 16), restricted to the 10 test adducts and good labels.
Class 3 (structure in no database) cannot be simulated; it scores 0 for everyone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .chem import TEST_ADDUCTS

NP_LIB = "enveda-np-examples"
TIMSTOF_LIB = "enveda-180"
NP_LIKE_LIBS = ("gnps", "riken", "msdial", "massbank", "mona", "pluskal_ms2", "spectraverse")
HOLDOUT_FRAC = 0.10
N_C2 = 1000
MAX_SET = 16
SPLIT_NAME = "solo-split-v1"


def draw_set_size(rng: np.random.Generator, available: int) -> int:
    return int(min(rng.geometric(0.25), MAX_SET, available))


def make_split(meta: pd.DataFrame, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    ok = meta.good_label & meta.adduct.isin(TEST_ADDUCTS)

    # c1: enveda-np-examples
    np_rows = meta[(meta.ingest_lib == NP_LIB)]
    np_keys = np.sort(np_rows.inchikey14.unique())

    # c2: hold out 10% of the other structures entirely
    others = np.sort(meta.loc[~meta.inchikey14.isin(np_keys), "inchikey14"].unique())
    holdout = set(rng.choice(others, size=int(HOLDOUT_FRAC * len(others)), replace=False))

    def sample_queries(rows: pd.DataFrame, keys, prefix: str, cls: str) -> list[pd.DataFrame]:
        out = []
        for k in keys:
            r = rows[rows.inchikey14 == k] if not isinstance(rows, dict) else rows[k]
            n = draw_set_size(rng, len(r))
            pick = r.iloc[rng.choice(len(r), size=n, replace=False)]
            out.append(pd.DataFrame({"molecule_id": f"{prefix}_{k}", "row_id": pick.row_id.to_numpy(),
                                     "inchikey14": k, "cls": cls, "source_lib": str(pick.ingest_lib.iloc[0])}))
        return out

    q1_rows = meta[(meta.ingest_lib == NP_LIB) & ok]
    q1 = sample_queries(dict(list(q1_rows.groupby("inchikey14", observed=True))),
                        [k for k in np_keys if k in set(q1_rows.inchikey14)], "c1", "c1")

    hold_rows = meta[meta.inchikey14.isin(holdout) & ok]
    q2 = []
    for libs, n in ((TIMSTOF_LIB,), N_C2 // 2), (NP_LIKE_LIBS, N_C2 - N_C2 // 2):
        r = hold_rows[hold_rows.ingest_lib.isin(libs)]
        # one source library per molecule: pick the structure, then its spectra from one library
        keys = np.sort(r.inchikey14.unique())
        keys = rng.choice(keys, size=min(n, len(keys)), replace=False)
        r = r[r.inchikey14.isin(set(keys))]
        by_key = {}
        for (k, _lib), g in r.groupby(["inchikey14", "ingest_lib"], observed=True):
            by_key.setdefault(k, []).append(g)
        groups = {k: v[rng.integers(len(v))] for k, v in by_key.items()}
        q2 += sample_queries(groups, sorted(groups), "c2", "c2")

    queries = pd.concat(q1 + q2, ignore_index=True)
    # rows that must never be used for training or library search
    banned = meta.inchikey14.isin(holdout) | (meta.ingest_lib == NP_LIB)
    return {"name": SPLIT_NAME, "seed": seed, "queries": queries, "holdout_keys": sorted(holdout),
            "c1_keys": list(np_keys), "banned_row": banned.to_numpy()}


SPLIT_V2 = "solo-split-v2"


def requery_c2(meta: pd.DataFrame, split_v1: dict, findable_keys: set, key_of: dict, seed: int = 2027) -> dict:
    """Split v2: same hold-outs and banned rows as v1; c2 queries re-drawn from findable hold-outs.

    By the competition's definition a class-2 molecule is in PubChem or COCONUT, so a c2 query must be a
    held-out structure present in our external candidate sources. Hold-outs that are in no external
    source behave like class 3 and are only counted for coverage. Within a structure, timsTOF spectra
    are used when available (the test is all timsTOF).
    """
    rng = np.random.default_rng(seed)
    ok = meta.good_label & meta.adduct.isin(TEST_ADDUCTS)
    hold = meta[meta.inchikey14.isin(split_v1["holdout_keys"]) & ok]
    hold = hold[hold.inchikey14.map(key_of).isin(findable_keys)]
    keys = np.sort(hold.inchikey14.unique())
    keys = set(rng.choice(keys, size=min(N_C2, len(keys)), replace=False))
    out = []
    for k, g in hold[hold.inchikey14.isin(keys)].groupby("inchikey14", observed=True):
        libs = g.ingest_lib.astype(str)
        lib = TIMSTOF_LIB if (libs == TIMSTOF_LIB).any() else rng.choice(sorted(libs.unique()))
        r = g[libs == lib]
        n = draw_set_size(rng, len(r))
        pick = r.iloc[rng.choice(len(r), size=n, replace=False)]
        out.append(pd.DataFrame({"molecule_id": f"c2_{k}", "row_id": pick.row_id.to_numpy(), "inchikey14": k,
                                 "cls": "c2", "source_lib": lib}))
    q1 = split_v1["queries"][split_v1["queries"].cls == "c1"]
    return {"queries": pd.concat([q1, *out], ignore_index=True), "holdout_keys": sorted(split_v1["holdout_keys"]),
            "banned_row": split_v1["banned_row"]}


def save_split(split: dict, folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    split["queries"].to_parquet(folder / "queries.parquet", index=False)
    pd.Series(split["holdout_keys"], name="inchikey14").to_frame().to_parquet(folder / "holdout_keys.parquet")
    np.save(folder / "banned_row.npy", split["banned_row"])


def load_split(folder) -> dict:
    return {"queries": pd.read_parquet(folder / "queries.parquet"),
            "holdout_keys": set(pd.read_parquet(folder / "holdout_keys.parquet").inchikey14),
            "banned_row": np.load(folder / "banned_row.npy")}
