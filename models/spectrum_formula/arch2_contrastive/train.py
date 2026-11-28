"""Contrastive spectrum-molecule model: training entry point. Owner: Member 4.

Team protocol (Member 3's split, pool, ranking helpers and scorer via ``common/team.py``). Each epoch embeds the
whole candidate pool and scores the validation molecules; the checkpoint with the best balanced MRR@25 is kept.
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
from common import contrastive as C  # noqa: E402
from common import features as F  # noqa: E402
from common import network as N  # noqa: E402
from common import retrieval as R  # noqa: E402
from common import team  # noqa: E402
from common.runtime import build_protocol, checkpoint_path, pick_device  # noqa: E402

from casmi.utils import load_config, set_seed  # noqa: E402


def build_model(cfg: dict, n_bits: int) -> C.ContrastiveNet:
    return C.build_model(cfg, n_bits)


def val_arrays(proto: team.Protocol, data_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    return team.spectra_arrays(proto.train_path, proto.queries, **F.spec_args(data_cfg))


def rank_split(
    model, proto: team.Protocol, queries, arrays, device, ppm: float, pool_bits=None, prior=None, prior_kwargs=None,
    retrieval: dict | None = None, data_cfg: dict | None = None, network: dict | None = None,
) -> dict:  # fmt: skip
    """Embed spectra and the whole pool, then rank each molecule's mass-window candidates by cosine.

    ``model`` may be a list: the models are then ensembled (average of ``scale * cosine`` over the models).
    ``prior``/``prior_kwargs`` (from ``priors.config_prior``) optionally add a documentation prior to the score;
    ``retrieval`` (``weight``, ``k``, ``beta``) optionally blends in analog-spectrum retrieval from the first model's
    embeddings (needs ``data_cfg``); ``network`` (see ``common/network.py``) optionally re-sorts each top-25 using the
    other query molecules' candidates.
    """
    ensemble = isinstance(model, list | tuple)
    models = list(model) if ensemble else [model]
    specs = [C.embed_spectra(m, *arrays, device) for m in models]
    aggregated = [C.aggregate_embeddings(queries.molecule_id, spec) for spec in specs]
    mol_ids, mol_embs = aggregated[0][0], [a[1] for a in aggregated]
    pool_embs = [C.embed_pool(m, proto.pool_fp, device, pool_bits=pool_bits) for m in models]
    scales = [float(m.scale.item()) for m in models]
    masses = proto.cp.consensus_masses(queries)
    extra = dict(prior_kwargs or {})
    if retrieval and retrieval.get("weight"):
        mols = queries.molecule_id.to_numpy()
        consensus = R.consensus_for(models[0], proto, specs[0], mols, data_cfg, retrieval, device)
        extra.update(consensus=consensus, retrieval_weight=float(retrieval["weight"]), pool_fp_packed=proto.pool_fp)
    rankings = C.rank_contrastive(
        proto.cp, proto.pool, pool_embs if ensemble else pool_embs[0], mol_ids, mol_embs if ensemble else mol_embs[0],
        masses, ppm, prior=prior, scale=scales if ensemble else scales[0], **extra,
    )  # fmt: skip
    return N.apply_network(rankings, mol_ids, mol_embs[0], proto.pool_fp, network)


def evaluate_mrr(model, proto: team.Protocol, arrays, device, ppm: float, pool_bits=None):
    rankings = rank_split(model, proto, proto.queries, arrays, device, ppm, pool_bits)
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
          f"device {device} ({time.time() - t0:.0f}s)")  # fmt: skip

    model = build_model(cfg, proto.cp.FP_BITS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps = tcfg["epochs"] * int(np.ceil(n / tcfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=steps)
    use_amp = device.type == "cuda" and tcfg.get("amp", True)  # half precision on CUDA only
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pool_mass, k, ppm_hard = proto.pool.mass.to_numpy(), tcfg["hard_negatives"], tcfg["hard_ppm"]
    rng = np.random.default_rng(cfg.get("seed", 2026))
    best, path = -1.0, checkpoint_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    pool_bits = C.pool_bits_on_device(proto.pool_fp, device)  # fingerprints stay on the device: no per-step CPU work

    def bits(idx: np.ndarray) -> torch.Tensor:
        return pool_bits[torch.from_numpy(idx).to(device)].float()

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        order, total = np.random.permutation(n), 0.0
        for i in range(0, n, tcfg["batch_size"]):
            idx = np.sort(order[i : i + tcfg["batch_size"]])
            pos = train_data["y_idx"][idx]
            hard = C.sample_hard_negatives(pool_mass, pos, k, ppm_hard, rng)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                spec = model.encode_spectrum(
                    torch.from_numpy(train_data["X"][idx].astype(np.float32)).to(device),
                    torch.from_numpy(train_data["meta"][idx]).to(device),
                )
                mol_pos = model.encode_molecule(bits(pos))
                mol_hard = model.encode_molecule(bits(hard.ravel())).view(len(idx), k, -1)
                pos_id, hard_id = torch.from_numpy(pos).to(device), torch.from_numpy(hard).to(device)
                loss = C.contrastive_loss(spec, mol_pos, mol_hard, pos_id, hard_id, model.scale)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        score, table = evaluate_mrr(model, proto, val, device, cfg["predict"]["ppm"], pool_bits)
        c1, c2 = table.loc["class 1", "MRR@25"], table.loc["class 2", "MRR@25"]
        print(f"epoch {epoch:2d}  train_loss {total / n:.4f}  scale {model.scale.item():.1f}  "
              f"val balanced MRR@25 {score:.4f}  (class 1 {c1:.4f}, class 2 {c2:.4f})")  # fmt: skip
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
