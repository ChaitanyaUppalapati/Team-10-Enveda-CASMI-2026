"""Quantized compact predictor (runtime check): writes CandidateScores for a split. Owner: Member 6."""

from __future__ import annotations

import argparse
from pathlib import Path

from casmi.interfaces import SPLITS, write_scores
from casmi.utils import load_config

HERE = Path(__file__).resolve().parent


def predict(cfg: dict, split: str):
    """Return a CandidateScores DataFrame (query_id, inchikey14, score) for ``split``."""
    raise NotImplementedError("TODO(Member 6): predict compact.arch2_quantized")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--split", choices=SPLITS, default="val")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    path = write_scores(predict(cfg, args.split), cfg["model_id"], args.split)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
