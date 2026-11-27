"""Train / predict logic shared by arch1_late_fusion and arch2_recall_priors (score-file mode).

Config keys (see each config.yaml): ``model_id``, ``inputs`` (model_ids to combine),
``truth`` (path to a parquet/CSV with query_id, inchikey14 of the true structure for the
development split), ``method`` / ``meta`` settings. Artefacts go to
``<output_dir>/models/<model_id>/``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from casmi.interfaces import QUERY_ID, TOP_K, rank_candidates
from casmi.utils.config import load_paths

from . import stacking
from .fusion import fuse, out_of_fold_fusion, tune_weights
from .inputs import load_inputs
from .metrics import mrr_at_k
from .normalize import normalize


def artefact_dir(cfg: dict) -> Path:
    path = Path(load_paths()["output_dir"]) / "models" / cfg["model_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_truth(cfg: dict) -> pd.DataFrame:
    path = cfg.get("truth")
    if not path:
        raise ValueError(
            "config.truth is empty: set it to the frozen validation labels "
            "(columns query_id, inchikey14) once Member 5 publishes the split."
        )
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def _top_k(df: pd.DataFrame) -> pd.DataFrame:
    return rank_candidates(df, top_k=TOP_K)[[QUERY_ID, "inchikey14", "score"]]


# ---- arch 1: late fusion -------------------------------------------------------------
def _oof_fusion(cfg: dict, norm: pd.DataFrame, truth: pd.DataFrame, ids: list[str]):
    return out_of_fold_fusion(norm, truth, ids, n_splits=cfg.get("n_splits", 5),
                              n_random=cfg.get("n_random", 200), seed=cfg.get("seed", 2026))


def train_fusion(cfg: dict) -> dict:
    """Tune weights on ``val`` and report them with an out-of-fold (cross-fitted) val MRR.

    ``in_sample_val_mrr`` uses weights fitted on the same queries, so it is optimistic and only for
    reference; ``oof_val_mrr`` is the number to report. The saved weights (fitted on all of ``val``)
    are the ones applied to ``test``.
    """
    ids, method = list(cfg["inputs"]), cfg.get("method", "rrf")
    wide, truth = load_inputs(ids, "val"), load_truth(cfg)
    norm = normalize(wide, ids, method)
    weights, in_sample = tune_weights(norm, truth, ids, n_random=cfg.get("n_random", 200),
                                      seed=cfg.get("seed", 2026))
    oof, fold_weights = _oof_fusion(cfg, norm, truth, ids)
    solo = {m: mrr_at_k(fuse(norm, {m: 1.0}), truth) for m in ids}
    report = {"method": method, "weights": weights, "oof_val_mrr": mrr_at_k(oof, truth),
              "in_sample_val_mrr": in_sample, "fold_weights": fold_weights, "single_model_val_mrr": solo}
    (artefact_dir(cfg) / "fusion.json").write_text(json.dumps(report, indent=2))
    return report


def predict_fusion(cfg: dict, split: str) -> pd.DataFrame:
    """CandidateScores for ``split``. For ``val`` these are out-of-fold scores, never the in-sample fit."""
    report = json.loads((artefact_dir(cfg) / "fusion.json").read_text())
    ids = list(report["weights"])
    norm = normalize(load_inputs(ids, split), ids, report["method"])
    if split == "val":
        # The saved weights were tuned on val labels; score each val query with weights from the other folds.
        return _top_k(_oof_fusion(cfg, norm, load_truth(cfg), ids)[0])
    return _top_k(fuse(norm, report["weights"]))


# ---- arch 2 score-file mode: stacked meta-learner-------------------------------------------------
def train_stack(cfg: dict) -> dict:
    ids, kind, seed = list(cfg["inputs"]), cfg.get("meta", "lgbm_rank"), cfg.get("seed", 2026)
    wide, truth = load_inputs(ids, "val"), load_truth(cfg)
    feats = stacking.build_features(wide, ids)
    y = stacking.add_labels(feats, truth)
    oof = stacking.out_of_fold_scores(feats, y, kind, cfg.get("n_splits", 5), seed)
    model, cols = stacking.fit_final(feats, y, kind, seed)
    with open(artefact_dir(cfg) / "stack.pkl", "wb") as fh:
        pickle.dump({"model": model, "cols": cols, "kind": kind, "inputs": ids}, fh)
    report = {"meta": kind, "oof_val_mrr": mrr_at_k(oof, truth), "n_features": len(cols)}
    (artefact_dir(cfg) / "stack.json").write_text(json.dumps(report, indent=2))
    return report


def predict_stack(cfg: dict, split: str) -> pd.DataFrame:
    with open(artefact_dir(cfg) / "stack.pkl", "rb") as fh:
        blob = pickle.load(fh)  # noqa: S301 - our own artefact, written by train_stack
    ids, kind = blob["inputs"], blob["kind"]
    feats = stacking.build_features(load_inputs(ids, split), ids)
    if split == "val":
        # The final model saw val labels, so report out-of-fold scores to avoid leakage.
        y = stacking.add_labels(feats, load_truth(cfg))
        return _top_k(stacking.out_of_fold_scores(feats, y, kind, cfg.get("n_splits", 5), cfg.get("seed", 2026)))
    return _top_k(stacking.score_with(blob["model"], blob["cols"], feats, kind))
