"""Day 2b: train S3 (spectrum -> fingerprint) and evaluate it on the frozen split with the honest+external pool.

Training rows: good-label library spectra of non-held-out structures with a test adduct, at most
``--per-structure`` spectra per structure (enveda-180 would otherwise dominate). 1% of the training
structures is kept aside for early stopping. Validation molecules are never seen.
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
from casmi_solo.candidates import load_external  # noqa: E402
from casmi_solo.chem import TEST_ADDUCTS  # noqa: E402
from casmi_solo.data import INTERIM, OUTPUTS, SpectrumReader, load_meta, load_structures  # noqa: E402
from casmi_solo.fingerprints import fingerprint_store, unpack  # noqa: E402
from casmi_solo.fpmodel import FPNet, batch_tensor, fp_cosine_scores, fp_loglik_scores, meta_features  # noqa: E402
from casmi_solo.fptrain import get_device, key_fold  # noqa: E402
from casmi_solo.scorer import evaluate  # noqa: E402
from casmi_solo.spectra import arrays_to_csr  # noqa: E402
from casmi_solo.split import SPLIT_NAME, load_split  # noqa: E402


def all_structures_table(structs, sources=("coconut", "pcl")):
    parts = [structs[["key", "smiles"]]] + [load_external(s)[["key", "smiles"]] for s in sources]
    return pd.concat(parts, ignore_index=True).dropna().drop_duplicates("key").reset_index(drop=True)


@torch.no_grad()
def predict(model, X, rows, precursor, meta, device, batch=2048):
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        out.append(model(batch_tensor(X, r, precursor[r], meta[r], device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2048), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-structure", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-minutes", type=float, default=120)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--name", default="s3_mlp_v1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-folds", type=int, default=1)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    stamp = lambda msg: print(f"[{(time.time() - t0) / 60:6.1f}m] {msg}", flush=True)  # noqa: E731
    out_dir = OUTPUTS / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    device = get_device()

    meta, structs = load_meta(), load_structures()
    split = load_split(INTERIM / "splits" / SPLIT_NAME)
    lib = Library.load_or_build(structs)
    fp, fp_index = fingerprint_store(all_structures_table(structs), "all")
    stamp(f"fingerprints: {len(fp_index):,} keys | device {device}")

    # ---- training rows (indices into the library matrix) ----
    lm = meta.iloc[lib.row_id]  # meta rows aligned with the library
    ok = (~split["banned_row"][lib.row_id]) & lm.adduct.isin(TEST_ADDUCTS).to_numpy() & pd.Series(
        lib.key).isin(fp_index.keys()).to_numpy()
    cand = pd.DataFrame({"i": np.flatnonzero(ok), "key": lib.key[ok]})
    cand = cand.sample(frac=1.0, random_state=a.seed).groupby("key").head(a.per_structure)
    keys = cand.key.unique()
    inner = set(rng.choice(keys, size=max(1, len(keys) // 100), replace=False))
    if a.n_folds > 1:  # out-of-fold model: structures of fold ``a.fold`` are never trained on
        cand = cand[key_fold(cand.key.to_numpy(), a.n_folds) != a.fold]
        inner = inner & set(cand.key)
    tr_rows = cand.i[~cand.key.isin(inner)].to_numpy()
    iv_rows = cand.i[cand.key.isin(inner)].to_numpy()[:20000]
    precursor = lm.precursor_mz.to_numpy()
    feats = meta_features(precursor, lm.ionization_mode.to_numpy() == "positive", lm.adduct.astype(str).to_numpy(),
                          lm.ce_mean.to_numpy(), lm.instrument_type.astype(str).to_numpy() == "timsTOF")
    target_idx = np.array([fp_index[k] for k in lib.key[np.concatenate([tr_rows, iv_rows])]])
    tgt_of = dict(zip(np.concatenate([tr_rows, iv_rows]), target_idx, strict=True))
    stamp(f"train {len(tr_rows):,} spectra / {len(keys) - len(inner):,} structures | inner val {len(iv_rows):,}")

    model = FPNet(width=a.width, depth=a.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr_rows) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()

    def targets(rows):
        return torch.from_numpy(unpack(fp[[tgt_of[r] for r in rows]]).astype(np.float32)).to(device)

    best, hist, step = np.inf, [], 0
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr_rows)
        tot, n = 0.0, 0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            r = perm[i:i + a.batch]
            loss = lossf(model(batch_tensor(lib.matrix, r, precursor[r], feats[r], device)), targets(r))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot, n = tot + loss.item(), n + 1
            if step % 500 == 0:
                stamp(f"  epoch {ep} step {step}/{steps} train BCE {tot / n:.4f}")
        lv = predict(model, lib.matrix, iv_rows, precursor, feats, device)
        iv_bce = float(lossf(torch.from_numpy(lv), targets(iv_rows).cpu()).item())
        hist.append({"epoch": ep, "train_bce": tot / max(n, 1), "inner_val_bce": iv_bce})
        stamp(f"epoch {ep}: train {tot / max(n, 1):.4f} | inner val {iv_bce:.4f}")
        if iv_bce < best:
            best = iv_bce
            torch.save(model.state_dict(), out_dir / "model.pt")
        if (time.time() - t0) / 60 > a.max_minutes:
            stamp("time budget reached")
            break
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    if a.no_eval:
        return

    # ---- validation on the frozen split ----
    q = split["queries"]
    mzs, ints = SpectrumReader().read(q.row_id.to_numpy())
    qm = meta.set_index("row_id").loc[q.row_id]
    Q = arrays_to_csr(mzs, ints, qm.precursor_mz.to_numpy())
    qprec = qm.precursor_mz.to_numpy()
    qfeat = meta_features(qprec, qm.ionization_mode.to_numpy() == "positive", qm.adduct.astype(str).to_numpy(),
                          qm.ce_mean.to_numpy(), qm.instrument_type.astype(str).to_numpy() == "timsTOF")
    logits = predict(model, Q, np.arange(len(q)), qprec, qfeat, device)
    np.save(out_dir / "val_logits.npy", logits)

    pool = Pool.from_table(pd.read_parquet(INTERIM / "pool_honest.parquet"))
    key_of = dict(zip(structs.inchikey14, structs.key, strict=True))
    truth = q.drop_duplicates("molecule_id")[["molecule_id", "inchikey14", "cls"]].assign(
        key=lambda d: d.inchikey14.map(key_of)).reset_index(drop=True)
    masses = pd.Series(qm.neutral_mass.to_numpy()).groupby(q.molecule_id.to_numpy()).median()
    pkeys = pool.table.key.to_numpy()
    ranks = {"S3_loglik": {}, "S3_cosine": {}}
    for mol, idx in pd.Series(np.arange(len(q))).groupby(q.molecule_id.to_numpy()):
        w = pool.window(masses[mol])
        if len(w) == 0:
            continue
        bits = unpack(fp[[fp_index.get(k, 0) for k in pkeys[w]]]).astype(np.float32)
        lm_ = logits[idx.to_numpy()].mean(0)
        for name, f in (("S3_loglik", fp_loglik_scores), ("S3_cosine", fp_cosine_scores)):
            ranks[name][mol] = pkeys[w[np.argsort(-f(lm_, bits), kind="stable")]].tolist()
    summary = {"history": hist}
    for name, r in ranks.items():
        tab, per_mol = evaluate(r, truth)
        per_mol.to_csv(out_dir / f"val_{name}.csv", index=False)
        summary[name] = tab.to_dict("index")
        stamp(f"{name}\n{tab.to_string()}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
