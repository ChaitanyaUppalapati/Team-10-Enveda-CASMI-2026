"""Command-line entry points shared only by Member 2's two architectures."""

import argparse

from casmi.interfaces import SPLITS, write_scores
from casmi.utils import load_config

from .runner import output_location, predict, train


def checked_config(path, architecture):
    cfg = load_config(path)
    if cfg["model"]["architecture"] != architecture:
        raise ValueError(f"This entry point requires architecture={architecture}")
    return cfg


def train_main(here, architecture, argv=None):
    parser = argparse.ArgumentParser(description=f"Train {architecture} on the configured prototype split")
    parser.add_argument("--config", default=here / "config.yaml")
    parser.add_argument("--smoke", action="store_true", help="One epoch, two batches per split; separate run directory")
    parser.add_argument("--run-dir", help="New or empty output directory (relative to repo root)")
    args = parser.parse_args(argv)
    print(f"run directory: {train(checked_config(args.config, architecture), run_dir=args.run_dir, smoke=args.smoke)}")


def predict_main(here, architecture, argv=None):
    parser = argparse.ArgumentParser(description=f"Score externally supplied candidates with {architecture}")
    parser.add_argument("--config", default=here / "config.yaml")
    parser.add_argument("--split", choices=SPLITS, default="val")
    parser.add_argument("--spectra", required=True, help="Parquet with explicit query_id and peak arrays")
    parser.add_argument("--candidates", required=True, help="Parquet with query_id, inchikey14, smiles")
    parser.add_argument("--checkpoint", help="Override imported notebook or module checkpoint")
    parser.add_argument(
        "--prototype", action="store_true", help="Write under outputs/prototype with a .prototype model ID"
    )
    parser.add_argument("--output-root", help="Override output root (prototype still adds its isolated subdirectory)")
    args = parser.parse_args(argv)
    cfg = checked_config(args.config, architecture)
    scores = predict(
        cfg,
        args.split,
        spectra_path=args.spectra,
        candidates_path=args.candidates,
        checkpoint_path=args.checkpoint,
        prototype=args.prototype,
    )
    root, model_id = output_location(cfg, args.prototype, args.output_root)
    print(f"wrote {write_scores(scores, model_id, args.split, root=root)}")
