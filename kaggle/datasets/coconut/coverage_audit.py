"""Candidate coverage audit on the training spectra. Owner: Member 4.

For each training library, treats every structure as one query (all its spectra, like a test
molecule), retrieves pool candidates by precursor mass and adduct, and reports how often the true
structure is among them. This is a PROVISIONAL proxy until the frozen validation split exists: the
structures here may also be in the pool because the pool and the libraries overlap by design.

Spectra are kept only if their adduct is one of the 10 test-set adducts and the recorded precursor
error is small, mirroring how the host cleaned the test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.candidates.coverage import coverage, pool_membership
from casmi.candidates.lookup import ADDUCT_DELTAS, candidates_for_queries
from casmi.candidates.pool import load_pool
from casmi.interfaces import INCHIKEY14, QUERY_ID
from casmi.utils.config import load_paths

COLUMNS = ["inchikey14", "precursor_mz", "adduct", "ingest_lib", "precursor_error_ppm"]


def load_queries(train_path: Path, max_error_ppm: float, per_library: int, seed: int) -> pd.DataFrame:
    """One query per (library, structure); spectra deduplicated by adduct and precursor m/z."""
    t = pd.read_parquet(train_path, columns=COLUMNS)
    t = t[t["adduct"].isin(ADDUCT_DELTAS) & (t["precursor_error_ppm"].abs() <= max_error_ppm)]
    t = t.assign(precursor_mz=t["precursor_mz"].round(4)).drop_duplicates([*COLUMNS[:4]])
    t[QUERY_ID] = t["ingest_lib"] + ":" + t["inchikey14"]
    keep = []
    for _, g in t.groupby("ingest_lib"):
        ids = g[QUERY_ID].drop_duplicates()
        keep.append(ids.sample(min(per_library, len(ids)), random_state=seed))
    return t[t[QUERY_ID].isin(pd.concat(keep))]


def audit(spectra: pd.DataFrame, pool: pd.DataFrame, ppms: tuple[float, ...]) -> pd.DataFrame:
    truth_all = spectra[[QUERY_ID, INCHIKEY14, "ingest_lib"]].drop_duplicates(QUERY_ID)
    rows = []
    cands = {ppm: candidates_for_queries(spectra, pool, ppm=ppm) for ppm in ppms}
    for lib, truth in [*truth_all.groupby("ingest_lib"), ("ALL", truth_all)]:
        row = {"library": lib, "n_structures": len(truth), "in_pool": pool_membership(pool, truth)}
        for ppm in ppms:
            row[f"coverage@{ppm:g}ppm"] = coverage(cands[ppm], truth)
        n = cands[ppms[0]].groupby(QUERY_ID).size().reindex(truth[QUERY_ID]).fillna(0)
        row[f"median_cands@{ppms[0]:g}ppm"] = float(np.median(n))
        row[f"p90_cands@{ppms[0]:g}ppm"] = float(np.quantile(n, 0.9))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="coconut.parquet", help="pool file name in candidates_dir")
    parser.add_argument("--train", type=Path, default=load_paths()["raw_dir"] / "train.parquet")
    parser.add_argument("--max-error-ppm", type=float, default=20.0)
    parser.add_argument("--per-library", type=int, default=2000, help="max structures sampled per library")
    parser.add_argument("--ppm", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    pool = load_pool(args.pool)
    spectra = load_queries(args.train, args.max_error_ppm, args.per_library, args.seed)
    n_q = spectra[QUERY_ID].nunique()
    print(f"pool {args.pool}: {len(pool):,} structures; {n_q:,} queries, {len(spectra):,} spectra")
    table = audit(spectra, pool, tuple(args.ppm))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
