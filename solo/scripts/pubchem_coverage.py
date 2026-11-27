"""How much class-2 coverage would a PubChem slice add, and at what pool size?

Streams PubChem Extras (CID-InChI-Key, CID-PMID, CID-Mass) once and caches per-CID arrays in
data/interim/pubchem_*.npy. Then, for literature-count thresholds, reports the slice size inside the mass
range and the share of split-v1 hold-out structures it contains (alone and together with COCONUT/PCL),
broken down by the hold-out structure's main library.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from casmi_solo.candidates import MASS_RANGE  # noqa: E402
from casmi_solo.data import INTERIM, REPO, load_meta, load_structures  # noqa: E402
from casmi_solo.split import SPLIT_NAME, load_split  # noqa: E402

PC = REPO / "data" / "external" / "pubchem"
CHUNK = 5_000_000


def stamp(t0, msg):
    print(f"[{(time.time() - t0) / 60:6.1f}m] {msg}", flush=True)


def read_chunks(name, usecols, dtype):
    return pd.read_csv(PC / name, sep="\t", header=None, usecols=usecols, dtype=dtype, chunksize=CHUNK,
                       quoting=3, on_bad_lines="skip", engine="c")


def pmid_counts(t0):
    out = INTERIM / "pubchem_pmid_count.npy"
    if out.exists():
        return np.load(out)
    counts = np.zeros(0, dtype=np.int32)
    for n, c in enumerate(read_chunks("CID-PMID.gz", [0], {0: np.int64})):
        b = np.bincount(c[0].to_numpy())
        if len(b) > len(counts):
            counts = np.pad(counts, (0, len(b) - len(counts)))
        counts[:len(b)] += b.astype(np.int32)
        if n % 10 == 0:
            stamp(t0, f"  PMID chunk {n}")
    np.save(out, counts)
    return counts


def masses(t0):
    out = INTERIM / "pubchem_mass.npy"
    if out.exists():
        return np.load(out)
    m = np.full(200_000_000, np.nan, dtype=np.float32)
    top = 0
    for n, c in enumerate(read_chunks("CID-Mass.gz", [0, 2], {0: np.int64, 2: np.float64})):
        cid = c[0].to_numpy()
        m[cid] = c[2].to_numpy()
        top = max(top, int(cid.max()) + 1)
        if n % 10 == 0:
            stamp(t0, f"  mass chunk {n}")
    np.save(out, m[:top])
    return m[:top]


def holdout_hits(t0, keys14):
    """CIDs whose standard InChIKey first block is a hold-out key: DataFrame(cid, inchikey14)."""
    out = INTERIM / "pubchem_holdout_hits.parquet"
    if out.exists():
        return pd.read_parquet(out)
    parts = []
    for n, c in enumerate(read_chunks("CID-InChI-Key.gz", [0, 2], {0: np.int64, 2: str})):
        k14 = c[2].str.slice(0, 14)
        hit = k14.isin(keys14)
        parts.append(pd.DataFrame({"cid": c[0][hit].to_numpy(), "inchikey14": k14[hit].to_numpy()}))
        if n % 5 == 0:
            stamp(t0, f"  InChIKey chunk {n}: {sum(len(p) for p in parts):,} hits")
    hits = pd.concat(parts, ignore_index=True)
    hits.to_parquet(out, index=False)
    return hits


def main() -> None:
    t0 = time.time()
    meta, structs = load_meta(), load_structures()
    split = load_split(INTERIM / "splits" / SPLIT_NAME)
    hold = sorted(split["holdout_keys"])
    lib_of = meta[meta.inchikey14.isin(hold)].groupby("inchikey14", observed=True).ingest_lib.agg(
        lambda s: s.astype(str).mode().iat[0])
    pool = pd.read_parquet(INTERIM / "pool_honest.parquet")
    key_of = dict(zip(structs.inchikey14, structs.key, strict=True))
    ext = set(pool.key[(pool.in_coconut + pool.in_pcl) > 0])
    hdf = pd.DataFrame({"inchikey14": hold})
    hdf["lib"] = hdf.inchikey14.map(lib_of).fillna("?")
    hdf["in_ext"] = hdf.inchikey14.map(key_of).isin(ext)

    counts = pmid_counts(t0)
    stamp(t0, f"PMID counts: {int((counts > 0).sum()):,} CIDs with literature")
    mass = masses(t0)
    stamp(t0, f"masses: {int(np.isfinite(mass).sum()):,} CIDs")
    hits = holdout_hits(t0, set(hold))
    stamp(t0, f"hold-out hits: {hits.inchikey14.nunique():,}/{len(hold):,} hold-out structures in PubChem")

    lo, hi = MASS_RANGE
    in_range = (mass >= lo) & (mass <= hi)
    cnt = np.zeros(len(mass), dtype=np.int32)
    cnt[:min(len(counts), len(mass))] = counts[:len(mass)]
    hits["pmid"] = cnt[np.minimum(hits.cid.to_numpy(), len(cnt) - 1)]
    best = hits.groupby("inchikey14").pmid.max()
    hdf["pc_pmid"] = hdf.inchikey14.map(best).fillna(-1).astype(int)

    rows = []
    for th in (None, 1, 2, 3, 5, 10, 25):
        if th is None:
            size, cov = int(in_range.sum()), hdf.pc_pmid >= 0
        else:
            size, cov = int((in_range & (cnt >= th)).sum()), hdf.pc_pmid >= th
        r = {"filter": "all PubChem" if th is None else f"pmid>={th}", "cids_in_mass_range": size,
             "pc_alone": cov.mean(), "pc_or_ext": (cov | hdf.in_ext).mean()}
        for lib, g in hdf.groupby("lib"):
            if len(g) >= 300:
                r[f"{lib}"] = (cov[g.index] | g.in_ext).mean()
        rows.append(r)
    base = {"filter": "COCONUT+PCL only", "cids_in_mass_range": 0, "pc_alone": 0.0, "pc_or_ext": hdf.in_ext.mean()}
    for lib, g in hdf.groupby("lib"):
        if len(g) >= 300:
            base[lib] = g.in_ext.mean()
    tab = pd.DataFrame([base, *rows]).set_index("filter")
    pd.set_option("display.width", 250)
    print(hdf.lib.value_counts().to_string())
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
