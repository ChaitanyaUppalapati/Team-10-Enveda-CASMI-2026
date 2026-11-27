"""Build the COCONUT candidate table for Kaggle. Owner: Member 4."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from casmi.candidates.coconut import COCONUT_FILE
from casmi.candidates.pool import build_pool

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="raw COCONUT export (csv/tsv/parquet)")
    parser.add_argument("--out", type=Path, default=HERE / "build")
    parser.add_argument("--workers", type=int, default=0, help="processes to use (0 = all cores)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pool = build_pool(args.source, workers=args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / COCONUT_FILE
    pool.to_parquet(target, index=False)
    print(f"wrote {len(pool):,} structures to {target}")


if __name__ == "__main__":
    main()
