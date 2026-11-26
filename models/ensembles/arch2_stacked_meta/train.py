"""Ensemble training entry point. Owner: Member 8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

from ensemble_common.runner import train_stack  # noqa: E402

from casmi.utils import load_config, set_seed  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2026))
    print(train_stack(cfg))


if __name__ == "__main__":
    main()
