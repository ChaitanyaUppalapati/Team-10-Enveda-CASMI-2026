"""Dual-branch spectrum + precursor network: writes CandidateScores for a split. Owner: Member 4.

``--split val`` prints the team protocol table (per-class and balanced MRR@25, Member 3's ``evaluate``).
``--split test`` also writes ``submission.csv`` next to the scores. Candidates: the team pool (training
structures) plus ``data.external_pools`` (COCONUT, PubChemLite, ...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import features as F  # noqa: E402
from common import team  # noqa: E402
from train import build_model, build_protocol, checkpoint_path, infer, pick_device, val_arrays  # noqa: E402

from casmi.interfaces import SPLITS, write_scores  # noqa: E402
from casmi.utils import load_config  # noqa: E402


def load_model(cfg: dict, n_bits: int, device):
    ckpt = torch.load(checkpoint_path(cfg), map_location=device)
    model = build_model(cfg, n_bits).to(device)
    model.load_state_dict(ckpt["state"])
    return model.eval()


def predict(cfg: dict, split: str, proto: team.Protocol | None = None):
    """Return ``(rankings, queries, truth_table_or_None, proto)`` for ``split`` (``val`` or ``test``)."""
    proto = proto or build_protocol(cfg)
    device = pick_device(cfg["train"].get("device", "auto"))
    model = load_model(cfg, proto.cp.FP_BITS, device)
    if split == "val":
        queries, truth = proto.queries, proto.truth
        arrays = val_arrays(proto, cfg["data"])
    elif split == "test":
        queries, truth = proto.test, None
        arrays = team.spectra_arrays(proto.test_path, proto.test, **F.spec_args(cfg["data"]))
    else:
        raise ValueError("predict supports the val and test splits")
    probs = infer(model, *arrays, device)
    rankings = proto.cp.rank_by_fingerprint(proto.pool, proto.pool_fp, queries, probs, ppm=cfg["predict"]["ppm"])
    return rankings, queries, truth, proto


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--split", choices=SPLITS, default="val")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    rankings, _, truth, proto = predict(cfg, args.split)
    if truth is not None:
        print(proto.cp.evaluate(rankings, truth, proto.pool.key.to_numpy()).to_string())
    path = write_scores(team.rankings_to_scores(rankings, proto.pool), cfg["model_id"], args.split)
    print(f"wrote {path}")
    if args.split == "test":
        out = path.parent / "submission.csv"
        proto.cp.write_submission(rankings, proto.pool, proto.test, path=str(out))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
