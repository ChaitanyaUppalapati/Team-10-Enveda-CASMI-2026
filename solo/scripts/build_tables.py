"""Build the cached metadata and structure tables in data/interim/."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi_solo.data import load_meta, load_structures  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    t0 = time.time()
    meta = load_meta(refresh=a.refresh)
    print(f"meta: {len(meta):,} spectra ({time.time() - t0:.0f}s)", flush=True)
    s = load_structures(meta, refresh=a.refresh, n_jobs=a.jobs)
    print(f"structures: {len(s):,} | key missing {s.key.isna().sum()} | key != inchikey14 "
          f"{(s.key != s.inchikey14).mean():.4f} | distinct keys {s.key.nunique():,} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
