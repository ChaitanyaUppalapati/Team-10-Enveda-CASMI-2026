"""Collect everything ``scripts/predict.py`` needs into one folder (to upload as a Kaggle Dataset).

The test-time pool and library use *all* training structures (hold-outs exist only for validation).
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from casmi_solo.candidates import build_pool_table  # noqa: E402
from casmi_solo.data import INTERIM, OUTPUTS, REPO, load_structures  # noqa: E402
from casmi_solo.fingerprints import extended_store, fingerprint_store  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fp import all_structures_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s3", default="s3_mlp_fold0,s3_mlp_fold1")
    ap.add_argument("--ranker", default="rerank_v2")
    ap.add_argument("--sources", default="coconut,pcl", help="external candidate sources in the pool")
    ap.add_argument("--dest", type=Path, default=REPO / "data" / "kaggle_artifacts")
    a = ap.parse_args()
    dest = a.dest
    (dest / "s3").mkdir(parents=True, exist_ok=True)

    structs = load_structures()
    sources = tuple(a.sources.split(","))
    pool = build_pool_table(structs, sources=sources)  # no exclusions
    pool.to_parquet(dest / "pool.parquet", index=False)
    fp, fp_index = fingerprint_store(all_structures_table(structs), "all")
    extra = tuple(s for s in sources if s not in ("coconut", "pcl"))
    if extra:
        fp, fp_index = extended_store((fp, fp_index), all_structures_table(structs, extra), "all")
    missing = set(pool.key) - set(fp_index)
    assert not missing, f"{len(missing)} pool keys without fingerprints"
    keep = np.array(sorted(fp_index[k] for k in pool.key))
    keys = np.array(list(fp_index), dtype=object)[keep]
    np.savez(dest / "fp_all.npz", keys=keys, fp=fp[keep])
    for old in (dest / "s3").glob("*/model.pt"):  # only the requested models ship
        if old.parent.name not in a.s3.split(","):
            shutil.rmtree(old.parent)
    shutil.copy(INTERIM / "library_v1.npz", dest / "library_v1.npz")
    for name in a.s3.split(","):
        (dest / "s3" / name).mkdir(exist_ok=True)
        shutil.copy(OUTPUTS / name / "model.pt", dest / "s3" / name / "model.pt")
    shutil.copy(OUTPUTS / a.ranker / "ranker.txt", dest / "ranker.txt")
    shutil.copytree(REPO / "solo" / "casmi_solo", dest / "code" / "casmi_solo", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "solo" / "scripts" / "predict.py", dest / "code" / "predict.py")
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1e9
    print(f"packaged {len(pool):,} candidates, {len(a.s3.split(','))} S3 model(s) -> {dest} ({total:.2f} GB)")


if __name__ == "__main__":
    main()
