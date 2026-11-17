"""Train fingerprint networks and score externally supplied candidate pairs."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from casmi.interfaces import INCHIKEY14_RE, validate_candidate_scores
from casmi.utils import set_seed
from casmi.utils.config import REPO_ROOT, load_paths

from .architectures import build_model
from .data import RDKIT_VERSION, SpectrumDataset, check_split, fingerprint, read_spectra


def resolve_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def member_dir(cfg):
    return resolve_path(os.environ.get("CASMI_MEMBER2_DIR") or cfg.get("member2_dir", "outputs/member2"))


def data_path(cfg, key):
    value = cfg["data"][key]
    path = Path(value).expanduser()
    return path if path.is_absolute() else member_dir(cfg) / path


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    model.train(optimizer is not None)
    samples, loss_sum, tp, fp, fn, tanimoto_sum = 0, 0.0, 0, 0, 0, 0.0
    for index, (x, y) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training/validation loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        samples += len(x)
        loss_sum += loss.item() * len(x)
        with torch.no_grad():
            pred, target = torch.sigmoid(logits) >= 0.5, y >= 0.5
            tp += (pred & target).sum().item()
            fp += (pred & ~target).sum().item()
            fn += (~pred & target).sum().item()
            union, intersection = (pred | target).sum(1), (pred & target).sum(1)
            tanimoto_sum += torch.where(union > 0, intersection / union.clamp(min=1), 1.0).sum().item()
    if samples == 0:
        raise ValueError("No batches processed")
    return {
        "loss": loss_sum / samples,
        "f1": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "tanimoto": tanimoto_sum / samples,
        "samples": samples,
    }


def train(cfg, *, run_dir=None, smoke=False):
    set_seed(cfg.get("seed", 42))
    settings, model_cfg = cfg["train"], cfg["model"]
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["patience"] < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    torch.set_num_threads(settings.get("cpu_threads", 2))
    device = device_for(cfg.get("device", "auto"))
    train_df = read_spectra(data_path(cfg, "train"), targets=True)
    val_df = read_spectra(data_path(cfg, "val"), targets=True)
    check_split(train_df, val_df, data_path(cfg, "split_file"))
    # Smoke uses two batches and a separate directory; imported weights are never overwritten.
    if smoke:
        train_df = train_df.iloc[: 2 * settings["batch_size"]]
        val_df = val_df.iloc[: 2 * settings["batch_size"]]
    counts = sum(
        (fingerprint(s, model_cfg["fp_size"], model_cfg["fp_radius"]).astype(np.int64) for s in train_df.smiles),
        np.zeros(model_cfg["fp_size"], dtype=np.int64),
    )
    weights = np.ones(model_cfg["fp_size"], dtype=np.float32)
    seen = counts > 0
    weights[seen] = np.clip((len(train_df) - counts[seen]) / counts[seen], 1, 20)
    train_loader = DataLoader(
        SpectrumDataset(train_df, model_cfg, targets=True),
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SpectrumDataset(val_df, model_cfg, targets=True), batch_size=settings["batch_size"], num_workers=0
    )
    model = build_model(model_cfg["architecture"], model_cfg["fp_size"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    output = (
        resolve_path(run_dir)
        if run_dir
        else member_dir(cfg) / "module_runs" / (cfg["model_id"] + (".smoke." if smoke else ".") + str(time.time_ns()))
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "model_id": cfg["model_id"],
        "mode": "prototype",
        "smoke": smoke,
        "seed": cfg.get("seed", 42),
        "rdkit_version": RDKIT_VERSION,
        "data": {key: str(data_path(cfg, key)) for key in ("train", "val", "split_file")},
        "config": cfg,
    }
    (output / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, (1 if smoke else settings["epochs"]) + 1):
        start = time.perf_counter()
        training = run_epoch(model, train_loader, criterion, device, optimizer)
        validation = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in training.items()},
            **{f"val_{k}": v for k, v in validation.items()},
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best - settings["min_delta"]:
            best, stale = validation["loss"], 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best,
                    "val_f1": validation["f1"],
                    "val_tanimoto": validation["tanimoto"],
                    "fp_size": model_cfg["fp_size"],
                    "mz_max": model_cfg["mz_max"],
                    "preprocessing": model_cfg,
                    "rdkit_version": RDKIT_VERSION,
                },
                output / "best.pt",
            )
        else:
            stale += 1
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if stale >= settings["patience"]:
            break
    return output


def load_model(cfg, checkpoint_path=None):
    model_cfg = cfg["model"]
    path = resolve_path(checkpoint_path) if checkpoint_path else member_dir(cfg) / cfg["predict"]["checkpoint"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["fp_size"] != model_cfg["fp_size"] or checkpoint["mz_max"] != model_cfg["mz_max"]:
        raise ValueError("Checkpoint fingerprint size or m/z range does not match config")
    if "preprocessing" in checkpoint:
        if checkpoint["preprocessing"] != model_cfg:
            raise ValueError("Checkpoint preprocessing does not match config")
    elif model_cfg["bin_width"] != 1.0 or model_cfg["fp_radius"] != 2:
        raise ValueError("Legacy notebook checkpoints require bin_width=1.0 and fp_radius=2")
    model = build_model(model_cfg["architecture"], model_cfg["fp_size"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device_for(cfg.get("device", "auto"))).eval(), path


def predict_probabilities(cfg, frame, checkpoint_path=None):
    torch.set_num_threads(cfg["train"].get("cpu_threads", 2))
    model, _ = load_model(cfg, checkpoint_path)
    loader = DataLoader(SpectrumDataset(frame, cfg["model"]), batch_size=cfg["predict"]["batch_size"], num_workers=0)
    probabilities = []
    with torch.inference_mode():
        for x in loader:
            probabilities.append(torch.sigmoid(model(x.to(next(model.parameters()).device))).cpu().numpy())
    return np.concatenate(probabilities)


def score_candidates(probabilities, query_ids, candidates, model_cfg):
    required = {"query_id", "inchikey14", "smiles"}
    if not required <= set(candidates.columns) or candidates.empty:
        raise ValueError("Candidate Parquet needs nonempty query_id, inchikey14, smiles columns")
    if candidates[list(required)].isnull().any().any():
        raise ValueError("Candidate rows contain nulls")
    if not candidates.inchikey14.astype(str).str.fullmatch(INCHIKEY14_RE.pattern).all():
        raise ValueError("Candidate inchikey14 must contain exactly 14 uppercase letters")
    if candidates.duplicated(["query_id", "inchikey14"]).any():
        raise ValueError("Candidate query/key pairs must be unique")
    if len(query_ids) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction rows")
    if set(query_ids) != set(candidates.query_id):
        raise ValueError("Candidate query IDs must exactly cover the input queries; empty queries are not dropped")
    # Multiple spectra for a real query are mean-pooled, as in the notebook.
    pooled = pd.DataFrame(probabilities).groupby(pd.Index(query_ids, name="query_id"), sort=False).mean()
    outputs = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        query = pooled.loc[query_id].to_numpy(dtype=np.float32)
        for start in range(0, len(group), 4096):
            chunk = group.iloc[start : start + 4096]
            fps = np.stack([fingerprint(s, model_cfg["fp_size"], model_cfg["fp_radius"]) for s in chunk.smiles])
            intersection = fps @ query
            denominator = query.sum() + fps.sum(axis=1) - intersection
            scores = intersection / np.maximum(denominator, 1e-8)
            outputs.append(
                pd.DataFrame({"query_id": query_id, "inchikey14": chunk.inchikey14.to_numpy(), "score": scores})
            )
    return validate_candidate_scores(pd.concat(outputs, ignore_index=True))


def predict(cfg, split, *, spectra_path=None, candidates_path=None, checkpoint_path=None, prototype=False):
    if not spectra_path or not candidates_path:
        raise ValueError("Provide --spectra and --candidates: candidates are never derived from validation labels")
    if not prototype and split == "val":
        raise ValueError(
            "Formal val is not wired to the team's frozen split yet. Use --prototype for prototype checks."
        )
    frame = read_spectra(resolve_path(spectra_path))
    if (
        "query_id" not in frame
        or frame.query_id.isnull().any()
        or not frame.query_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    ):
        raise ValueError(
            "Spectra require explicit, nonempty string query_id values; label-derived IDs are not inferred"
        )
    candidates = pd.read_parquet(resolve_path(candidates_path))
    if "smiles" not in candidates and "normalized_smiles" in candidates:
        candidates = candidates.rename(columns={"normalized_smiles": "smiles"})
    return score_candidates(
        predict_probabilities(cfg, frame, checkpoint_path), frame.query_id, candidates, cfg["model"]
    )


def output_location(cfg, prototype, root=None):
    base = resolve_path(root) if root else load_paths()["output_dir"]
    return (base / "prototype", cfg["model_id"] + ".prototype") if prototype else (base, cfg["model_id"])
