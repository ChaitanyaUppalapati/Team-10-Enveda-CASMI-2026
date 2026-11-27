"""Day 2a: process COCONUT + PubChemLite, then measure candidate coverage and B1/B2 on the honest pool."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from casmi_solo.baselines import Library, Pool  # noqa: E402
from casmi_solo.candidates import build_pool_table, load_external  # noqa: E402
from casmi_solo.data import INTERIM, OUTPUTS, SpectrumReader, load_meta, load_structures  # noqa: E402
from casmi_solo.scorer import evaluate  # noqa: E402
from casmi_solo.spectra import arrays_to_csr  # noqa: E402
from casmi_solo.split import SPLIT_NAME, load_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sources", default="coconut,pcl")
    ap.add_argument("--pool-out", default="pool_honest.parquet")
    ap.add_argument("--out", default="day2")
    a = ap.parse_args()
    t0 = time.time()
    stamp = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)  # noqa: E731
    out_dir = OUTPUTS / a.out
    sources = tuple(a.sources.split(","))
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in sources:
        e = load_external(s, refresh=a.refresh)
        stamp(f"{s}: {len(e):,} unique keys, mass {e.mass.min():.0f}-{e.mass.max():.0f}")

    meta, structs = load_meta(), load_structures()
    split = load_split(INTERIM / "splits" / SPLIT_NAME)
    q = split["queries"]
    key_of = dict(zip(structs.inchikey14, structs.key, strict=True))
    truth = q.drop_duplicates("molecule_id")[["molecule_id", "inchikey14", "cls"]].assign(
        key=lambda d: d.inchikey14.map(key_of)).reset_index(drop=True)

    table = build_pool_table(structs, split["holdout_keys"], sources)
    pool = Pool.from_table(table)
    src_cols = ["in_train", *(f"in_{s}" for s in sources)]
    stamp(f"honest pool: {len(pool.table):,} keys | " + " | ".join(f"{c} {int(table[c].sum()):,}" for c in src_cols))

    # coverage: is the answer anywhere in the pool, and from which source?
    t = table.set_index("key")
    cov = truth.assign(in_pool=truth.key.isin(t.index))
    for c in src_cols:
        cov[c] = truth.key.map(t[c]).fillna(0).astype(bool)
    cov["lib"] = q.drop_duplicates("molecule_id").source_lib.to_numpy()
    cov_table = cov.groupby("cls")[["in_pool", *src_cols]].mean().round(3)
    stamp(f"c2 answer coverage by source library:\n"
          f"{cov[cov.cls == 'c2'].groupby('lib')[['in_pool', *src_cols]].mean().round(3).to_string()}")
    stamp(f"answer coverage by class:\n{cov_table.to_string()}")
    # also: coverage of the real NP examples (test-like chemistry) by external sources alone
    np_keys = set(structs.loc[structs.inchikey14.isin(split.get('c1_keys', [])), "key"]) or set(
        truth.loc[truth.cls == "c1", "key"])
    ext = set(table.key[table[src_cols[1:]].sum(axis=1) > 0])
    stamp(f"NP examples (c1) found in COCONUT or PubChemLite: {np.mean([k in ext for k in np_keys]):.3f}")

    lib = Library.load_or_build(structs)
    mzs, ints = SpectrumReader().read(q.row_id.to_numpy())
    qm = meta.set_index("row_id").loc[q.row_id]
    Q = arrays_to_csr(mzs, ints, qm.precursor_mz.to_numpy())
    spectra = pd.DataFrame({"molecule_id": q.molecule_id.to_numpy(), "neutral_mass": qm.neutral_mass.to_numpy(),
                            "mode": (qm.ionization_mode.to_numpy() == "positive").astype(np.int8)})
    allowed = ~split["banned_row"][lib.row_id]
    n_cand = [len(pool.window(m)) for m in spectra.groupby("molecule_id").neutral_mass.median()]
    stamp(f"candidates per molecule within 10 ppm: median {np.median(n_cand):.0f}, 90% {np.quantile(n_cand, .9):.0f}")
    summary = {"coverage": cov_table.to_dict("index")}
    for meth, r in run(pool, lib, spectra, Q, allowed).items():
        tab, per_mol = evaluate(r, truth)
        per_mol.to_csv(out_dir / f"val_{meth}_ext.csv", index=False)
        summary[meth] = tab.to_dict("index")
        stamp(f"{meth} pool=honest+external\n{tab.to_string()}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    table.to_parquet(INTERIM / a.pool_out, index=False)


if __name__ == "__main__":
    main()
