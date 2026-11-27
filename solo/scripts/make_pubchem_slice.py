"""Write data/external/pubchem/pubchem_slice.csv: PubChem compounds with >= --min-pmid PubMed references
inside the pre-filter mass range, with SMILES, monoisotopic mass and the reference count.

Needs the per-CID arrays cached by scripts/pubchem_coverage.py. Afterwards build the candidates with
``load_external("pc", refresh=True)`` (scripts/build_candidates.py).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from casmi_solo.candidates import MASS_RANGE, PREFILTER_MAX, SOURCES  # noqa: E402
from casmi_solo.data import INTERIM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubchem_coverage import PC, read_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-pmid", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    counts, mass = np.load(INTERIM / "pubchem_pmid_count.npy"), np.load(INTERIM / "pubchem_mass.npy")
    n = min(len(counts), len(mass))
    keep = np.zeros(len(mass), dtype=bool)
    keep[:n] = (counts[:n] >= a.min_pmid) & (mass[:n] >= MASS_RANGE[0]) & (mass[:n] <= PREFILTER_MAX)
    print(f"{int(keep.sum()):,} CIDs selected", flush=True)
    parts = []
    for c in read_chunks("CID-SMILES.gz", [0, 1], {0: np.int64, 1: str}):
        cid = c[0].to_numpy()
        sel = keep[np.minimum(cid, len(keep) - 1)] & (cid < len(keep))
        parts.append(pd.DataFrame({"cid": cid[sel], "SMILES": c[1].to_numpy()[sel]}))
    out = pd.concat(parts, ignore_index=True)
    out["mass"] = mass[out.cid.to_numpy()]
    out["pmid_count"] = counts[out.cid.to_numpy()]
    dest = SOURCES["pc"][0]
    out.to_csv(dest, index=False)
    print(f"wrote {len(out):,} rows -> {dest} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    assert PC.exists()
    main()
