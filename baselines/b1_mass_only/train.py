"""B1: mass-only baseline: training entry point. Owner: Member 3."""

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
    raise NotImplementedError("TODO(Member 3): train baselines.b1_mass_only")


if __name__ == "__main__":
    main()
