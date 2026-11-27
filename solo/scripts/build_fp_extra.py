"""Fingerprints for external-source keys missing from fp_all.npz (cached as fp_all_extra.npz)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from casmi_solo.data import load_structures  # noqa: E402
from casmi_solo.fingerprints import extended_store, fingerprint_store  # noqa: E402
from train_fp import all_structures_table  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default="pc")
    a = ap.parse_args()
    s = load_structures()
    fp, idx = extended_store(fingerprint_store(all_structures_table(s), "all"),
                             all_structures_table(s, tuple(a.sources.split(","))), "all")
    print(f"fingerprints: {fp.shape}, {len(idx):,} keys")
