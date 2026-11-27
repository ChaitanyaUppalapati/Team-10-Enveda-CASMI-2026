"""Build validation split v2 (findable class-2 queries) and score B1/B2 on it with the honest pool."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from casmi_solo.baselines import Library, Pool  # noqa: E402
from casmi_solo.data import INTERIM, OUTPUTS, SpectrumReader, load_meta, load_structures  # noqa: E402
from casmi_solo.scorer import evaluate  # noqa: E402
from casmi_solo.spectra import arrays_to_csr  # noqa: E402
from casmi_solo.split import SPLIT_NAME, SPLIT_V2, load_split, requery_c2, save_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import run  # noqa: E402


def main() -> None:
    meta, structs = load_meta(), load_structures()
    table = pd.read_parquet(INTERIM / "pool_honest.parquet")
    findable = set(table.key[(table.in_coconut + table.in_pcl) > 0])
    key_of = dict(zip(structs.inchikey14, structs.key, strict=True))
    v1 = load_split(INTERIM / "splits" / SPLIT_NAME)
    hold_keys = {key_of[k] for k in v1["holdout_keys"]}
    print(f"hold-outs findable in external sources: {np.mean([k in findable for k in hold_keys]):.3f} "
          f"of {len(hold_keys):,}")
    v2 = requery_c2(meta, v1, findable, key_of)
    save_split(v2, INTERIM / "splits" / SPLIT_V2)
    q = v2["queries"]
    m = q.drop_duplicates("molecule_id")
    print(f"split v2: {m.cls.value_counts().to_dict()} molecules, {len(q)} spectra | c2 source libs "
          f"{m[m.cls == 'c2'].source_lib.value_counts().to_dict()}")

    lib, pool = Library.load_or_build(structs), Pool.from_table(table)
    mzs, ints = SpectrumReader().read(q.row_id.to_numpy())
    qm = meta.set_index("row_id").loc[q.row_id]
    Q = arrays_to_csr(mzs, ints, qm.precursor_mz.to_numpy())
    spectra = pd.DataFrame({"molecule_id": q.molecule_id.to_numpy(), "neutral_mass": qm.neutral_mass.to_numpy(),
                            "mode": (qm.ionization_mode.to_numpy() == "positive").astype(np.int8)})
    truth = m[["molecule_id", "inchikey14", "cls"]].assign(key=lambda d: d.inchikey14.map(key_of)).reset_index(
        drop=True)
    out_dir = OUTPUTS / "split_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for meth, r in run(pool, lib, spectra, Q, ~v2["banned_row"][lib.row_id]).items():
        tab, per_mol = evaluate(r, truth)
        per_mol.to_csv(out_dir / f"val_{meth}.csv", index=False)
        summary[meth] = tab.to_dict("index")
        print(meth, "\n", tab.to_string())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
