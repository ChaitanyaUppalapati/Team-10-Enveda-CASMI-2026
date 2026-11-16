"""Build the COCONUT candidate table for Kaggle. Owner: Member 4."""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="raw COCONUT export")
    parser.add_argument("--out", type=Path, default=HERE / "build")
    parser.parse_args(argv)
    raise NotImplementedError("TODO(Member 4): standardise structures, compute inchikey14/mass")


if __name__ == "__main__":
    main()
