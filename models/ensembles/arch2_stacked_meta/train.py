"""Stacked meta-learner over all models: training entry point. Owner: Member 8."""

from __future__ import annotations

import argparse
from pathlib import Path

from casmi.utils import load_config, set_seed

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2026))
    raise NotImplementedError("TODO(Member 8): train ensembles.arch2_stacked_meta")


if __name__ == "__main__":
    main()
