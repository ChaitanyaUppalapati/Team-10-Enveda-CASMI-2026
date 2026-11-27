"""Dual-branch spectrum + precursor network: training entry point. Owner: Member 4.

Runs on the team protocol (Member 3's split, pool, ranking and scorer, via ``common/team.py``) and keeps
the checkpoint with the best balanced validation MRR@25.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
import torch.nn.functional as nnf  # noqa: E402
from common import features as F  # noqa: E402
from common import team  # noqa: E402
from common.model import DualBranchNet  # noqa: E402
from common.runtime import build_protocol, checkpoint_path, pick_device  # noqa: E402

from casmi.utils import load_config, set_seed  # noqa: E402


@torch.no_grad()
def infer(model, X: np.ndarray, meta: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    """Per-spectrum bit probabilities ``(n, n_bits)``."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        logits, _ = model(
            torch.from_numpy(X[i : i + batch].astype(np.float32)).to(device),
            torch.from_numpy(meta[i : i + batch]).to(device),
        )
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def val_arrays(proto: team.Protocol, data_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    return team.spectra_arrays(proto.train_path, proto.queries, **F.spec_args(data_cfg))


def build_model(cfg: dict, n_bits: int) -> DualBranchNet:
    return DualBranchNet(n_bins=F.input_dim(**F.spec_args(cfg["data"])), n_bits=n_bits, **cfg["model"])


def evaluate_mrr(model, proto: team.Protocol, arrays, device, ppm: float):
    """Balanced MRR@25 and the full per-class table on the team protocol."""
    probs = infer(model, *arrays, device)
    rankings = proto.cp.rank_by_fingerprint(proto.pool, proto.pool_fp, proto.queries, probs, ppm=ppm)
    table = proto.cp.evaluate(rankings, proto.truth, proto.pool.key.to_numpy())
    return float(table.loc["balanced", "MRR@25"]), table


def train(cfg: dict, proto: team.Protocol | None = None) -> Path:
    set_seed(cfg.get("seed", 2026))
    tcfg, data_cfg = cfg["train"], cfg["data"]
    device = pick_device(tcfg.get("device", "auto"))
    t0 = time.time()
    proto = proto or build_protocol(cfg)
    train_data, val = team.training_data(proto, data_cfg), val_arrays(proto, data_cfg)
    n = len(train_data["y_idx"])
    print(f"protocol: {n:,} train spectra, {len(proto.truth):,} val molecules, pool {len(proto.pool):,} structures, "
          f"device {device} ({time.time() - t0:.0f}s)")
    f_targets = team.formula_targets(proto.pool)

    model = build_model(cfg, proto.cp.FP_BITS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps = tcfg["epochs"] * int(np.ceil(n / tcfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=steps)
    best, path = -1.0, checkpoint_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        order, total = np.random.permutation(n), 0.0
        for i in range(0, n, tcfg["batch_size"]):
            idx = np.sort(order[i : i + tcfg["batch_size"]])
            y_idx = train_data["y_idx"][idx]
            x = torch.from_numpy(train_data["X"][idx].astype(np.float32)).to(device)
            m = torch.from_numpy(train_data["meta"][idx]).to(device)
            y = torch.from_numpy(np.unpackbits(proto.pool_fp[y_idx], axis=1).astype(np.float32)).to(device)
            f = torch.from_numpy(f_targets[y_idx]).to(device)
            logits, formula = model(x, m)
            loss = nnf.binary_cross_entropy_with_logits(logits, y)
            loss = loss + tcfg["formula_loss_weight"] * nnf.smooth_l1_loss(formula, f)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        score, table = evaluate_mrr(model, proto, val, device, cfg["predict"]["ppm"])
        c1, c2 = table.loc["class 1", "MRR@25"], table.loc["class 2", "MRR@25"]
        print(f"epoch {epoch:2d}  train_loss {total / n:.4f}  val balanced MRR@25 {score:.4f}  "
              f"(class 1 {c1:.4f}, class 2 {c2:.4f})")
        if score > best:
            best = score
            torch.save({"state": model.state_dict(), "cfg": cfg}, path)
    print(f"best balanced MRR@25 {best:.4f} -> {path} ({time.time() - t0:.0f}s total)")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = parser.parse_args(argv)
    train(load_config(args.config))


if __name__ == "__main__":
    main()
