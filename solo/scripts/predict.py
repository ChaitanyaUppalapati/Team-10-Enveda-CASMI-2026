"""Inference: test.parquet -> submission.csv with the full candidate pool, full library, S3 and the reranker.

Runs locally or in the offline Kaggle notebook. Everything heavy is precomputed and read from
``--artifacts`` (built by ``scripts/package_artifacts.py``):

    pool.parquet         full candidate table (training structures + COCONUT + PubChemLite), one row per key
    fp_all.npz           packed Morgan fingerprints for every pool key
    library_v1.npz       binned library spectra (row -> key, mode)
    s3/<name>/model.pt   one or more S3 models (logits are averaged)
    ranker.txt           LightGBM reranker

Molecules whose window is empty fall back to a wider mass window, then to placeholder SMILES, so the file
always has 25 guesses per molecule.
"""

import argparse
import sys
import time
from pathlib import Path

for _p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):  # packaged / repo layout
    sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
from casmi_solo.baselines import Library, Pool  # noqa: E402
from casmi_solo.chem import neutral_mass  # noqa: E402
from casmi_solo.fpmodel import FPNet  # noqa: E402
from casmi_solo.fptrain import get_device, predict, query_features  # noqa: E402
from casmi_solo.rerank import FEATURES, molecule_features  # noqa: E402
from casmi_solo.scorer import TOP_K, write_submission  # noqa: E402
from casmi_solo.spectra import table_to_csr  # noqa: E402


def load_library(path: Path) -> Library:
    z = np.load(path, allow_pickle=True)
    m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return Library(m, z["row_id"], z["key"], z["mode"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    a = ap.parse_args()
    t0 = time.time()
    stamp = lambda msg: print(f"[{(time.time() - t0) / 60:6.1f}m] {msg}", flush=True)  # noqa: E731
    device = get_device()
    art = a.artifacts

    pool = Pool.from_table(pd.read_parquet(art / "pool.parquet"))
    z = np.load(art / "fp_all.npz", allow_pickle=True)
    fp, fp_index = z["fp"], {k: i for i, k in enumerate(z["keys"])}
    lib = load_library(art / "library_v1.npz")
    models = []
    for p in sorted((art / "s3").glob("*/model.pt")):
        m = FPNet().to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        models.append(m)
    import lightgbm as lgb

    ranker = lgb.Booster(model_file=str(art / "ranker.txt"))
    stamp(f"loaded pool {len(pool.table):,}, library {lib.matrix.shape[0]:,}, {len(models)} S3 model(s)")

    test = pq.read_table(a.test).to_pandas()
    test["neutral_mass"] = neutral_mass(test.precursor_mz.to_numpy(), test.adduct.to_numpy())
    test["ce_mean"] = test.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
    Q = table_to_csr(pq.read_table(a.test, columns=["ms2_mzs", "ms2_normalized_intensities"]),
                     test.precursor_mz.to_numpy())
    prec, feats = query_features(test)
    rows = np.arange(len(test))
    logits = np.mean([predict(m, Q, rows, prec, feats, device) for m in models], axis=0)
    modes = (test.ionization_mode.to_numpy() == "positive").astype(np.int8)
    stamp(f"test: {test.molecule_id.nunique()} molecules, {len(test)} spectra")

    smiles_of = dict(zip(pool.table.key, pool.table.smiles, strict=True))
    out = {}
    for mol, idx in pd.Series(rows).groupby(test.molecule_id.to_numpy()):
        i = idx.to_numpy()
        mass = float(np.nanmedian(test.neutral_mass.to_numpy()[i]))
        ranked = []
        if np.isfinite(mass):
            for ppm in (10.0, 30.0):
                f = molecule_features(mol, mass, pool, lib, Q[i], modes[i], None, logits[i], fp, fp_index, ppm=ppm)
                if len(f):
                    f = f.assign(s=ranker.predict(f[FEATURES])).sort_values("s", ascending=False)
                    ranked = f.key.tolist()
                    if len(ranked) >= TOP_K:
                        break
        out[mol] = [smiles_of[k] for k in ranked[:TOP_K]]
    sub = write_submission(out, test.molecule_id.unique(), a.out)
    n_full = int((sub.smiles.str.count(";") + 1 == TOP_K).sum())
    stamp(f"wrote {a.out}: {len(sub)} molecules, {n_full} with {TOP_K} guesses")


if __name__ == "__main__":
    main()
