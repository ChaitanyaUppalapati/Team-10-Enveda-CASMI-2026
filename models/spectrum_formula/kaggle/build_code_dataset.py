"""Stage the Kaggle code dataset for the spectrum_formula notebook. Owner: Member 4.

Keeps the repo layout (``src/casmi``, ``configs``, ``models/spectrum_formula``, Member 3's pipeline) so
``casmi.utils.config.REPO_ROOT`` and ``common/team.py`` resolve paths exactly as they do locally.

    python models/spectrum_formula/kaggle/build_code_dataset.py --kaggle-user <handle> --out <dir>
    kaggle datasets create -p <dir>            # first upload (keep the dataset private)
    kaggle datasets version -p <dir> -m "msg"  # later updates
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SLUG = "casmi-m4-code"
MARKER = "CASMI_M4_CODE"  # the notebook finds the dataset by globbing for this file
PIPELINES = ("baselines/b1_mass_only/casmi_pipeline.py", "models/spectrum_encoders/common/casmi_pipeline.py")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "build", "notebooks", "results", "tests")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-user", required=True)
    parser.add_argument("--out", type=Path, default=HERE / "build")
    args = parser.parse_args(argv)

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(REPO / "src" / "casmi", args.out / "src" / "casmi", ignore=IGNORE)
    shutil.copytree(REPO / "configs", args.out / "configs")
    shutil.copytree(REPO / "models" / "spectrum_formula", args.out / "models" / "spectrum_formula", ignore=IGNORE)
    pipeline = next((REPO / p for p in PIPELINES if (REPO / p).is_file()), None)
    if pipeline is None:
        raise FileNotFoundError(f"casmi_pipeline.py not found in {PIPELINES}")
    target = args.out / pipeline.relative_to(REPO)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(pipeline, target)
    m7 = REPO / "models" / "spectra_pooling" / "common" / "candidates.py"  # Member 7's gate, used by b2gate.py
    (args.out / m7.relative_to(REPO)).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(m7, args.out / m7.relative_to(REPO))
    (args.out / MARKER).write_text("marker file the Kaggle notebook searches for\n")
    meta = {"title": SLUG, "id": f"{args.kaggle_user}/{SLUG}", "licenses": [{"name": "other"}], "isPrivate": True}
    (args.out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"staged {sum(1 for p in args.out.rglob('*') if p.is_file())} files in {args.out}")


if __name__ == "__main__":
    main()
