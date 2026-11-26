"""Full pooling run on Member 3's pipeline: embeddings, pooled ranking, fair control, export.

Member 3's ``casmi_pipeline`` (``baselines/b1_mass_only``) provides the validation split
(classes 1 and 2), the candidate pool, the per-spectrum fingerprint model and the
evaluation. This module adds what pooling needs on top of it. Owner: Member 7.

Pooling methods compared on the same frozen per-spectrum model:

- ``mean``: average of per-spectrum probabilities (what ``rank_by_fingerprint`` does).
- ``mean_tuned``: fair control. The per-spectrum head gets the same extra training as the
  poolers, with the loss on the averaged probabilities, but no spectrum sees another.
  If the poolers only beat ``mean`` and not ``mean_tuned``, the gain is extra training,
  not pooling.
- ``deepsets`` (arch2) and ``attention`` (arch1).
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .pooling import build_pool
from .sets import pad_sets, sample_sets

METHODS = ["mean", "mean_tuned", "deepsets", "attention"]
KAGGLE_METHODS = ["mean", "mean_tuned", "attention_head"]


# ---------- per-spectrum model (Member 3's ResidualMLP) ----------


def embed_batch(model, mz, intensity, precursor, adduct):
    """Hidden vector h of ResidualMLP, i.e. everything before ``model.head``."""
    bins = mz.long().clamp(0, model.n_bins - 1)
    x = torch.zeros(mz.size(0), model.n_bins, device=mz.device, dtype=intensity.dtype)
    x.scatter_add_(1, bins, intensity)
    x = torch.cat([x, (precursor / 1000.0).unsqueeze(1), model.adduct(adduct)], dim=1)
    return model.blocks(model.stem(x))


@torch.no_grad()
def embed_all(model, mz, intensity, precursor, adduct, device, batch=2048, dtype=torch.float16):
    """Embeddings for many spectra, returned on the CPU (they can be millions of rows)."""
    model.eval()
    out = []
    for i in range(0, len(mz), batch):
        sl = slice(i, i + batch)
        h = embed_batch(model,
                        torch.as_tensor(mz[sl], device=device),
                        torch.as_tensor(intensity[sl], device=device),
                        torch.as_tensor(precursor[sl], dtype=torch.float32, device=device),
                        torch.as_tensor(adduct[sl], device=device))
        out.append(h.to(dtype).cpu())
    return torch.cat(out)


def unpack_bits(packed: torch.Tensor) -> torch.Tensor:
    """(B, F/8) uint8 -> (B, F) float, MSB first like np.packbits."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> shifts) & 1).flatten(1).float()


# ---------- stage B: pooling heads and the fair control ----------


class TunedMean(torch.nn.Module):
    """Control: per-spectrum head, then the average of probabilities. Same call signature as the pools."""

    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.head = copy.deepcopy(head)
        for p in self.head.parameters():  # the encoder is frozen; this copy is trained
            p.requires_grad_(True)

    def forward(self, h, mask):
        probs = torch.sigmoid(self.head(h).float())
        m = mask.unsqueeze(-1).to(probs.dtype)
        mean = (probs * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.logit(mean.clamp(1e-6, 1 - 1e-6)), None


class AttentionHead(torch.nn.Module):
    """Attention that keeps the trained per-spectrum head.

    The full-run notebook showed DeepSets/attention lose because they replace the trained
    head with a fresh one. Here attention only decides how much each spectrum's
    probabilities count; the head itself is a trainable copy, exactly as in ``TunedMean``.
    With uniform weights this is ``TunedMean``.
    """

    def __init__(self, head: torch.nn.Module, emb: int, att: int = 128):
        super().__init__()
        self.head = copy.deepcopy(head)
        for p in self.head.parameters():
            p.requires_grad_(True)
        self.V = torch.nn.Linear(emb, att)
        self.U = torch.nn.Linear(emb, att)
        self.w = torch.nn.Linear(att, 1)
        torch.nn.init.zeros_(self.w.weight)  # start as the plain average
        torch.nn.init.zeros_(self.w.bias)

    def forward(self, h, mask):
        scores = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        weights = torch.softmax(scores.float().masked_fill(~mask, float("-inf")), dim=1)
        probs = torch.sigmoid(self.head(h).float())
        mean = (weights.unsqueeze(-1) * probs).sum(1)
        return torch.logit(mean.clamp(1e-6, 1 - 1e-6)), weights


def make_pool(name, emb, fp_bits, head=None):
    if name == "mean_tuned":
        return TunedMean(head)
    if name == "attention_head":
        return AttentionHead(head, emb)
    return build_pool(name, emb, fp_bits)


def batched_forward(pool, sets, H, device, chunk=1024):
    """Run a pool over many sets. H lives on the CPU; each chunk is gathered then moved."""
    logits, weights = [], []
    for s in range(0, len(sets), chunk):
        h, mask = pad_sets(sets[s:s + chunk], H)
        lg, w = pool(h.to(device).float(), mask.to(device))
        logits.append(lg.float())
        if w is not None:
            weights += [w[i, :len(sets[s + i])].detach().cpu().numpy() for i in range(len(w))]
    return torch.cat(logits), (weights or None)


def train_pool(pool, H, groups, train_keys, target_fp, device, H_val, val_sets, val_bits,
               epochs=20, batch=256, lr=1e-3, seed=0, log=print, ckpt=None):
    """Train one pooling head on test-like random subsets of each training molecule's spectra.

    H: train embeddings (CPU). target_fp: dict key -> packed fingerprint (uint8).
    H_val, val_sets, val_bits: fixed validation sets and their (n_sets, F) 0/1 targets.
    Keeps the epoch with the lowest validation BCE.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pool = pool.to(device)
    opt = torch.optim.AdamW(pool.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best, best_state, hist = float("inf"), None, []
    for ep in range(epochs):
        pool.train()
        order = rng.permutation(len(train_keys))
        total, n = 0.0, 0
        for s in range(0, len(order), batch):
            keys = [train_keys[j] for j in order[s:s + batch]]
            h, mask = pad_sets(sample_sets(groups, keys, rng), H)
            logits, _ = pool(h.to(device).float(), mask.to(device))
            target = unpack_bits(torch.as_tensor(np.stack([target_fp[k] for k in keys]), device=device))
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pool.parameters(), 1.0)
            opt.step()
            total, n = total + loss.item(), n + 1
        sched.step()
        pool.eval()
        with torch.no_grad():
            logits, _ = batched_forward(pool, val_sets, H_val, device)
            vl = float(F.binary_cross_entropy_with_logits(logits, val_bits.to(device)))
        hist.append({"epoch": ep + 1, "train_bce": total / max(n, 1), "val_bce": vl})
        log(f"  epoch {ep + 1}: train {total / max(n, 1):.4f}  val {vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.detach().cpu().clone() for k, v in pool.state_dict().items()}
    pool.load_state_dict(best_state)
    if ckpt:
        torch.save(best_state, ckpt)
    return pool.eval(), pd.DataFrame(hist)


# ---------- ranking with Member 3's pool and evaluation ----------


def rank_pooled(cp, pool, pool_fp_packed, masses, probs, ppm=10.0):
    """Rank candidates for each molecule by soft Tanimoto to ONE pooled probability vector.

    Same as ``cp.rank_by_fingerprint`` except the pooling has already happened.
    masses: Series molecule_id -> neutral mass. probs: (n_molecules, F) in the same order.
    """
    pool_mass = pool.mass.to_numpy()
    rankings = {}
    for (mol, mass), p in zip(masses.items(), probs, strict=True):
        idx = cp.window_indices(pool_mass, mass, ppm)
        if len(idx):
            bits = np.unpackbits(pool_fp_packed[idx], axis=1).astype(np.float32)
            overlap = bits @ p
            tanimoto = overlap / (p.sum() + bits.sum(axis=1) - overlap + 1e-8)
            idx = idx[np.argsort(-tanimoto, kind="stable")]
        rankings[mol] = cp.fill_to_k(idx, pool_mass, mass)
    return rankings


def rrf_fuse(rank_lists, weights=None, k=60, top=25):
    """Reciprocal-rank fusion of several ranked lists of pool indices (best first)."""
    weights = weights or [1.0] * len(rank_lists)
    score = {}
    for lst, w in zip(rank_lists, weights, strict=True):
        for r, idx in enumerate(lst):
            score[int(idx)] = score.get(int(idx), 0.0) + w / (k + r + 1)
    return sorted(score, key=lambda i: (-score[i], i))[:top]


def paired_bootstrap(rr_a, rr_b, n=2000, seed=3):
    """Mean difference a - b in reciprocal rank with a 95% CI (same molecules, resampled together)."""
    diff = np.asarray(rr_a, float) - np.asarray(rr_b, float)
    if len(diff) == 0:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    draws = np.random.default_rng(seed).choice(diff, size=(n, len(diff))).mean(axis=1)
    return {"delta": float(diff.mean()), "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975))}


# ---------- export for the offline Kaggle notebook (Member 6) ----------


def save_artifact(folder, encoder_kwargs, encoder_state, method, pool_state, emb, fp_bits, settings):
    """Everything needed to rank test molecules offline, except the candidate pool files."""
    os.makedirs(folder, exist_ok=True)
    torch.save(encoder_state, os.path.join(folder, "encoder.pt"))
    torch.save(pool_state, os.path.join(folder, f"pool_{method}.pt"))
    info = {"encoder": "residual_mlp", "encoder_kwargs": encoder_kwargs, "method": method,
            "emb": emb, "fp_bits": fp_bits, "settings": settings}
    with open(os.path.join(folder, "pooling_artifact.json"), "w") as f:
        json.dump(info, f, indent=2)


def load_artifact(folder, cp, device, head_for_control=None):
    with open(os.path.join(folder, "pooling_artifact.json")) as f:
        info = json.load(f)
    enc = cp.build_models()[info["encoder"]](**info["encoder_kwargs"])
    enc.load_state_dict(torch.load(os.path.join(folder, "encoder.pt"), map_location="cpu"))
    enc = enc.to(device).eval()
    pool = make_pool(info["method"], info["emb"], info["fp_bits"], head=enc.head)
    pool.load_state_dict(torch.load(os.path.join(folder, f"pool_{info['method']}.pt"), map_location="cpu"))
    return enc, pool.to(device).eval(), info


@torch.no_grad()
def predict_molecules(cp, enc, pool, spectra, mz, intensity, device):
    """Pooled fingerprint probabilities per molecule.

    spectra: rows with molecule_id, precursor_mz, adduct, neutral_mass, aligned with mz/intensity.
    Returns (masses Series molecule_id -> median neutral mass, probs array in the same order).
    """
    H = embed_all(enc, mz, intensity, spectra.precursor_mz.to_numpy(), cp.adduct_ids(spectra.adduct), device)
    pos = pd.Series(np.arange(len(spectra)), index=spectra.index)
    groups = spectra.assign(pos=pos.to_numpy()).groupby("molecule_id", sort=False)
    sets = [g.pos.to_numpy() for _, g in groups]
    masses = groups.neutral_mass.median()
    if isinstance(pool, str) and pool == "mean":
        probs = []
        for s in range(0, len(sets), 1024):
            h, mask = pad_sets(sets[s:s + 1024], H)
            p = torch.sigmoid(enc.head(h.to(device).float()).float())
            m = mask.to(device).unsqueeze(-1).float()
            probs.append(((p * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
        return masses, torch.cat(probs).numpy()
    logits, _ = batched_forward(pool, sets, H, device)
    return masses, torch.sigmoid(logits).cpu().numpy()
