"""Build the Kaggle code dataset: casmi wheel + configs + frozen split + selected weights.

Owner: Member 6.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "build")
    parser.add_argument("--weights", type=Path, nargs="*", default=[],
                        help="weight folders to include (copied under weights/)")
    args = parser.parse_args(argv)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(args.out),
                    str(REPO)], check=True)
    shutil.copytree(REPO / "configs", args.out / "configs")
    shutil.copytree(REPO / "data" / "splits", args.out / "splits")
    for w in args.weights:
        shutil.copytree(w, args.out / "weights" / w.name)
    shutil.copy(HERE / "dataset-metadata.json", args.out / "dataset-metadata.json")
    print(f"dataset ready in {args.out}")


if __name__ == "__main__":
    main()
