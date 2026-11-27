"""Score-file mode of arch2: stacked meta-learner CandidateScores for a split. Owner: Member 8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

from ensemble_common.runner import predict_stack  # noqa: E402

from casmi.interfaces import SPLITS, write_scores  # noqa: E402
from casmi.utils import load_config  # noqa: E402


def predict(cfg: dict, split: str):
    """Return a CandidateScores DataFrame (query_id, inchikey14, score) for ``split``."""
    return predict_stack(cfg, split)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--split", choices=SPLITS, default="val")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(f"wrote {write_scores(predict(cfg, args.split), cfg['model_id'], args.split)}")


if __name__ == "__main__":
    main()
