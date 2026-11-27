"""Day 1: frozen split, spectral library, B1 + B2 on validation, and a B2 test submission.

Validation reports two pools:
- ``honest``: training structures minus the c2 hold-outs (what we really have before external candidates)
- ``oracle``: every training structure (c2 answers always present: separates ranking from coverage)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from casmi_solo.baselines import Library, Pool, molecule_mass, rank_b1, rank_b2, train_pool_frame  # noqa: E402
from casmi_solo.chem import neutral_mass  # noqa: E402
from casmi_solo.data import INTERIM, OUTPUTS, TEST, SpectrumReader, load_meta, load_structures  # noqa: E402
from casmi_solo.scorer import evaluate, write_submission  # noqa: E402
from casmi_solo.spectra import arrays_to_csr, table_to_csr  # noqa: E402
from casmi_solo.split import SPLIT_NAME, load_split, make_split, save_split  # noqa: E402


def run(pool, lib, spectra, Q, allowed, methods=("B1", "B2")):
    """spectra: one row per query spectrum (molecule_id, neutral_mass, mode); Q: CSR aligned with it."""
    masses = molecule_mass(spectra)
    out = {m: {} for m in methods}
    keys = pool.table.key.to_numpy()
    for mol, g in spectra.groupby("molecule_id", sort=False):
        mass = masses[mol]
        if "B1" in methods:
            out["B1"][mol] = keys[rank_b1(pool, mass)].tolist()
        if "B2" in methods:
            idx = g.index.to_numpy()
            r, _ = rank_b2(pool, lib, mass, Q[idx], g["mode"].to_numpy(), allowed)
            out["B2"][mol] = keys[r].tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-split", action="store_true")
    ap.add_argument("--refresh-library", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    stamp = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)  # noqa: E731
    out_dir = OUTPUTS / "day1"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, structs = load_meta(), load_structures()
    split_dir = INTERIM / "splits" / SPLIT_NAME
    if a.refresh_split or not split_dir.exists():
        save_split(make_split(meta), split_dir)
    split = load_split(split_dir)
    q = split["queries"]
    stamp(f"split: {q.molecule_id.nunique()} molecules ({q.groupby('cls').molecule_id.nunique().to_dict()}), "
          f"{len(q)} spectra, {len(split['holdout_keys']):,} held-out structures")

    lib = Library.load_or_build(structs, refresh=a.refresh_library)
    stamp(f"library: {lib.matrix.shape[0]:,} spectra, {lib.matrix.nnz:,} peaks")

    # validation query spectra
    mzs, ints = SpectrumReader().read(q.row_id.to_numpy())
    qm = meta.set_index("row_id").loc[q.row_id]
    Q = arrays_to_csr(mzs, ints, qm.precursor_mz.to_numpy())
    spectra = pd.DataFrame({"molecule_id": q.molecule_id.to_numpy(), "neutral_mass": qm.neutral_mass.to_numpy(),
                            "mode": (qm.ionization_mode.to_numpy() == "positive").astype(np.int8)})
    key_of = dict(zip(structs.inchikey14, structs.key, strict=True))
    truth = q.drop_duplicates("molecule_id")[["molecule_id", "inchikey14", "cls"]].assign(
        key=lambda d: d.inchikey14.map(key_of)).reset_index(drop=True)
    allowed = ~split["banned_row"][lib.row_id]

    pools = {"honest": Pool.build([train_pool_frame(structs, split["holdout_keys"])]),
             "oracle": Pool.build([train_pool_frame(structs)])}
    results = {}
    for pname, pool in pools.items():
        ranks = run(pool, lib, spectra, Q, allowed)
        for meth, r in ranks.items():
            table, per_mol = evaluate(r, truth)
            results[f"{meth}/{pname}"] = table
            per_mol.to_csv(out_dir / f"val_{meth}_{pname}.csv", index=False)
            stamp(f"{meth} pool={pname}\n{table.to_string()}")
    summary = {k: v.to_dict("index") for k, v in results.items()}
    (out_dir / "val_summary.json").write_text(json.dumps(summary, indent=2))

    # test submission (placeholder test; Kaggle swaps in the hidden set): full pool, full library
    test = pq.read_table(TEST).to_pandas()
    te_spec = pd.DataFrame({"molecule_id": test.molecule_id,
                            "neutral_mass": neutral_mass(test.precursor_mz.to_numpy(), test.adduct.to_numpy()),
                            "mode": (test.ionization_mode.to_numpy() == "positive").astype(np.int8)})
    Qt = table_to_csr(pq.read_table(TEST, columns=["ms2_mzs", "ms2_normalized_intensities"]),
                      test.precursor_mz.to_numpy())
    full = pools["oracle"]
    ranks = run(full, lib, te_spec, Qt, None, methods=("B2",))["B2"]
    smiles_of = dict(zip(full.table.key, full.table.smiles, strict=True))
    sub = write_submission({m: [smiles_of[k] for k in ks[:25]] for m, ks in ranks.items()},
                           test.molecule_id.unique(), out_dir / "submission_b2.csv")
    stamp(f"test submission: {len(sub)} molecules -> {out_dir / 'submission_b2.csv'}")


if __name__ == "__main__":
    main()
