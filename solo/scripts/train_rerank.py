"""Day 3: LightGBM LambdaRank reranker over B1/B2/S3/prior features, evaluated on split v2.

Reranker training queries never overlap the validation molecules, and none of them was seen by S3:
- c2-like: findable held-out structures not used by split v2 (as in validation: library has none of them)
- c1-like: S3's inner early-stopping structures; the query library's spectra of each structure are
  removed from the search library, its spectra in other libraries stay (as for the NP examples)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from casmi_solo.baselines import Library, Pool  # noqa: E402
from casmi_solo.chem import TEST_ADDUCTS  # noqa: E402
from casmi_solo.data import INTERIM, OUTPUTS, SpectrumReader, load_meta, load_structures  # noqa: E402
from casmi_solo.fingerprints import extended_store, fingerprint_store  # noqa: E402
from casmi_solo.fpmodel import FPNet  # noqa: E402
from casmi_solo.fptrain import get_device, key_fold, predict, query_features, training_rows  # noqa: E402
from casmi_solo.rerank import FEATURES, molecule_features, rankings_from_scores, train_ranker  # noqa: E402
from casmi_solo.scorer import evaluate  # noqa: E402
from casmi_solo.spectra import arrays_to_csr  # noqa: E402
from casmi_solo.split import SPLIT_V2, TIMSTOF_LIB, draw_set_size, load_split  # noqa: E402

EXT_COLS = ("in_coconut", "in_pcl", "in_pc")  # external sources a pool table may carry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fp import all_structures_table  # noqa: E402


def sample_molecule(rng, g, prefix, k, cls, lib=None):
    libs = g.ingest_lib.astype(str)
    if lib is None:
        lib = TIMSTOF_LIB if (libs == TIMSTOF_LIB).any() else rng.choice(sorted(libs.unique()))
    r = g[libs == lib]
    pick = r.iloc[rng.choice(len(r), size=draw_set_size(rng, len(r)), replace=False)]
    return pd.DataFrame({"molecule_id": f"{prefix}_{k}", "row_id": pick.row_id.to_numpy(), "inchikey14": k,
                         "cls": cls, "source_lib": lib})


def make_logit_fn(models, device, key_of, oof: bool):
    """S3 logits per query spectrum. With out-of-fold models, a ``t1_`` (c1-like training) query is scored
    by the fold model that never trained on its structure; everything else by the mean of all models."""

    def fn(queries, Q, prec, feats):
        rows = np.arange(len(queries))
        all_l = np.stack([predict(m, Q, rows, prec, feats, device) for m in models])
        out = all_l.mean(0)
        if oof:
            is_t1 = queries.molecule_id.str.startswith("t1_").to_numpy()
            if is_t1.any():
                folds = key_fold(queries.inchikey14.map(key_of).to_numpy()[is_t1], len(models))
                out[is_t1] = all_l[folds, np.flatnonzero(is_t1)]
        return out

    return fn


def build_features(queries, meta, lib, pool, allowed, logit_fn, fp, fp_index, key_of, stamp):
    mzs, ints = SpectrumReader().read(queries.row_id.to_numpy())
    qm = meta.set_index("row_id").loc[queries.row_id]
    Q = arrays_to_csr(mzs, ints, qm.precursor_mz.to_numpy())
    prec, feats = query_features(qm)
    logits = logit_fn(queries, Q, prec, feats)
    modes = (qm.ionization_mode.to_numpy() == "positive").astype(np.int8)
    mass = pd.Series(qm.neutral_mass.to_numpy()).groupby(queries.molecule_id.to_numpy()).median()
    truth = queries.drop_duplicates("molecule_id").set_index("molecule_id").inchikey14.map(key_of)
    frames = []
    groups = pd.Series(np.arange(len(queries))).groupby(queries.molecule_id.to_numpy())
    for n, (mol, idx) in enumerate(groups):
        i = idx.to_numpy()
        f = molecule_features(mol, mass[mol], pool, lib, Q[i], modes[i], allowed, logits[i], fp, fp_index)
        if len(f):
            f["label"] = f.key == truth[mol]
            frames.append(f)
        if (n + 1) % 1000 == 0:
            stamp(f"  features: {n + 1:,}/{len(groups):,} molecules")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s3", default="s3_mlp_v1", help="one model dir, or comma-separated out-of-fold model dirs")
    ap.add_argument("--n-c2", type=int, default=2500)
    ap.add_argument("--n-c1", type=int, default=1200)
    ap.add_argument("--per-structure", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="rerank_v1")
    ap.add_argument("--pool", default="pool_honest.parquet", help="honest pool table in data/interim")
    ap.add_argument("--split", default=SPLIT_V2)
    ap.add_argument("--sources", default="coconut,pcl", help="external sources whose keys need fingerprints")
    a = ap.parse_args()
    t0 = time.time()
    stamp = lambda msg: print(f"[{(time.time() - t0) / 60:6.1f}m] {msg}", flush=True)  # noqa: E731
    out_dir = OUTPUTS / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    meta, structs = load_meta(), load_structures()
    key_of = dict(zip(structs.inchikey14, structs.key, strict=True))
    lib = Library.load_or_build(structs)
    table = pd.read_parquet(INTERIM / a.pool)
    pool = Pool.from_table(table)
    fp, fp_index = fingerprint_store(all_structures_table(structs), "all")
    extra = [s for s in a.sources.split(",") if s not in ("coconut", "pcl")]
    if extra:
        fp, fp_index = extended_store((fp, fp_index), all_structures_table(structs, extra), "all")
    split = load_split(INTERIM / "splits" / a.split)
    ext_cols = [c for c in EXT_COLS if c in table]
    banned = split["banned_row"]
    names = a.s3.split(",")
    oof = len(names) > 1
    models = []
    for n in names:
        m = FPNet().to(device)
        m.load_state_dict(torch.load(OUTPUTS / n / "model.pt", map_location=device))
        models.append(m)
    logit_fn = make_logit_fn(models, device, key_of, oof)
    if oof:  # any non-held-out structure can be a c1-like query (scored by the model that never saw it)
        c1_pool = set(structs.key[~structs.inchikey14.isin(split["holdout_keys"])]) & set(
            table.key[table.in_coconut > 0])  # natural products, like the test set
    else:
        _, _, c1_pool, _ = training_rows(meta, lib, banned, fp_index.keys(), a.per_structure, a.seed)
    stamp(f"loaded: pool {len(pool.table):,}, S3 models {names}, c1-like candidate structures {len(c1_pool):,}")

    # ---- reranker training queries ----
    rng = np.random.default_rng(a.seed + 7)
    ok = meta.good_label & meta.adduct.isin(TEST_ADDUCTS)
    val_keys = set(split["queries"].inchikey14)
    findable = set(table.key[table[ext_cols].sum(axis=1) > 0])
    hold = meta[meta.inchikey14.isin(split["holdout_keys"]) & ok & ~meta.inchikey14.isin(val_keys)]
    hold = hold[hold.inchikey14.map(key_of).isin(findable)]
    c2k = rng.choice(np.sort(hold.inchikey14.unique()), size=min(a.n_c2, hold.inchikey14.nunique()), replace=False)
    parts = [sample_molecule(rng, g, "t2", k, "c2") for k, g in
             hold[hold.inchikey14.isin(set(c2k))].groupby("inchikey14", observed=True)]

    c1_14 = set(structs.inchikey14[structs.key.isin(c1_pool)])
    cand = meta[meta.inchikey14.isin(c1_14) & ok & ~banned]
    multi = cand.groupby("inchikey14", observed=True).ingest_lib.nunique()
    c1k = rng.choice(np.sort(multi.index[multi >= 2]), size=min(a.n_c1, int((multi >= 2).sum())), replace=False)
    pairs = []
    for k, g in cand[cand.inchikey14.isin(set(c1k))].groupby("inchikey14", observed=True):
        lib_name = rng.choice(sorted(g.ingest_lib.astype(str).unique()))
        parts.append(sample_molecule(rng, g, "t1", k, "c1", lib_name))
        pairs.append((k, lib_name))
    # the query library's spectra of each c1-like structure leave the search library
    pair_set = set(pairs)
    sub = meta[meta.inchikey14.isin(set(c1k))]
    hit = [(k, str(lb)) in pair_set for k, lb in zip(sub.inchikey14, sub.ingest_lib, strict=True)]
    extra_ban = np.zeros(len(meta), dtype=bool)
    extra_ban[sub.row_id.to_numpy()[np.array(hit, dtype=bool)]] = True
    tq = pd.concat(parts, ignore_index=True)
    stamp(f"reranker training queries: {tq.drop_duplicates('molecule_id').cls.value_counts().to_dict()}")

    allowed_train = ~(banned | extra_ban)[lib.row_id]
    allowed_val = ~banned[lib.row_id]
    ftr = build_features(tq, meta, lib, pool, allowed_train, logit_fn, fp, fp_index, key_of, stamp)
    fva = build_features(split["queries"], meta, lib, pool, allowed_val, logit_fn, fp, fp_index, key_of, stamp)
    ftr.to_parquet(out_dir / "features_train.parquet", index=False)
    fva.to_parquet(out_dir / "features_val.parquet", index=False)
    stamp(f"features: train {len(ftr):,} rows ({ftr.label.sum()} positives), val {len(fva):,} rows")

    # keep only molecules whose answer is in the window (others carry no ranking signal)
    has_pos = ftr.groupby("molecule_id").label.transform("any")
    ranker = train_ranker(ftr[has_pos])
    ranker.save_model(str(out_dir / "ranker.txt"))
    imp = pd.Series(ranker.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    stamp(f"feature importance (gain, top 12):\n{(imp / imp.sum()).round(3).head(12).to_string()}")

    truth = split["queries"].drop_duplicates("molecule_id")[["molecule_id", "inchikey14", "cls"]].assign(
        key=lambda d: d.inchikey14.map(key_of)).reset_index(drop=True)
    scores = {"B1_prior": fva.prior.to_numpy() - 1e-3 * fva.ppm_abs.to_numpy(),
              "B2": np.where(fva.b2 >= 0, 10 + fva.b2, fva.prior - 1e-3 * fva.ppm_abs),
              "S3_ll": fva.s3_ll.to_numpy(), "S3_cos": fva.s3_cos.to_numpy(),
              "rerank": ranker.predict(fva[FEATURES])}
    summary = {}
    for name, s in scores.items():
        tab, per_mol = evaluate(rankings_from_scores(fva, s), truth)
        per_mol.to_csv(out_dir / f"val_{name}.csv", index=False)
        summary[name] = tab.to_dict("index")
        stamp(f"{name}\n{tab.to_string()}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
