"""Contrastive spectrum-molecule model: encoders, loss, hard negatives, ranking. Owner: Member 4.

A spectrum encoder (same inputs as arch1: binned spectrum, optional neutral-loss block, precursor/adduct/CE
features) and a molecule encoder (Morgan bits) map into one unit-norm space. Training uses InfoNCE with
in-batch negatives plus hard negatives: pool structures whose mass is within a few ppm of the true structure,
which are exactly the candidates the model must separate at test time. Candidates are ranked by cosine
similarity to the molecule's averaged spectrum embedding.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnf
from torch import nn

from . import features as F


class ContrastiveNet(nn.Module):
    def __init__(
        self,
        n_bins: int = F.N_BINS,
        n_meta: int = F.N_META,
        n_bits: int = 2048,
        d: int = 512,
        spec_dim: int = 1024,
        meta_dim: int = 64,
        hidden: int = 2048,
        mol_hidden: int = 2048,
        dropout: float = 0.2,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.spectrum = nn.Sequential(
            nn.Linear(n_bins, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, spec_dim)
        )
        self.meta = nn.Sequential(nn.Linear(n_meta, meta_dim), nn.GELU(), nn.Linear(meta_dim, meta_dim))
        self.trunk = nn.Sequential(
            nn.GELU(), nn.Linear(spec_dim + meta_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )  # fmt: skip
        self.molecule = nn.Sequential(
            nn.Linear(n_bits, mol_hidden), nn.LayerNorm(mol_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mol_hidden, d),
        )  # fmt: skip
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def encode_spectrum(self, spectrum: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        h = self.trunk(torch.cat([self.spectrum(spectrum), self.meta(meta)], dim=-1))
        return nnf.normalize(h, dim=-1)

    def encode_molecule(self, bits: torch.Tensor) -> torch.Tensor:
        return nnf.normalize(self.molecule(bits), dim=-1)

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100.0)


def contrastive_loss(
    spec: torch.Tensor,
    mol_pos: torch.Tensor,
    mol_hard: torch.Tensor,
    pos_id: torch.Tensor,
    hard_id: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """InfoNCE over the batch's other molecules and each spectrum's hard negatives.

    ``spec``/``mol_pos``: (B, d) unit vectors; ``mol_hard``: (B, K, d); ``pos_id``: (B,) structure ids;
    ``hard_id``: (B, K). Other batch items with the same structure and hard negatives equal to the positive
    are masked out, so the same molecule is never treated as a negative of itself.
    """
    spec, mol_pos, mol_hard = spec.float(), mol_pos.float(), mol_hard.float()  # logits in fp32 even under autocast
    b = spec.shape[0]
    logits_batch = scale * spec @ mol_pos.T
    same = (pos_id[:, None] == pos_id[None, :]) & ~torch.eye(b, dtype=torch.bool, device=spec.device)
    logits_batch = logits_batch.masked_fill(same, float("-inf"))
    logits_hard = scale * torch.einsum("bd,bkd->bk", spec, mol_hard)
    logits_hard = logits_hard.masked_fill(hard_id == pos_id[:, None], float("-inf"))
    logits = torch.cat([logits_batch, logits_hard], dim=1)
    return nnf.cross_entropy(logits, torch.arange(b, device=spec.device))


def sample_hard_negatives(
    pool_mass: np.ndarray, pos_idx: np.ndarray, k: int, ppm: float, rng: np.random.Generator
) -> np.ndarray:
    """``(B, k)`` pool indices of structures within ``ppm`` of each positive's mass (``pool_mass`` is sorted).

    When a positive has no same-mass neighbour, negatives come from its neighbours in mass order instead.
    """
    mass = pool_mass[pos_idx]
    lo = np.searchsorted(pool_mass, mass * (1 - ppm * 1e-6), side="left")
    hi = np.searchsorted(pool_mass, mass * (1 + ppm * 1e-6), side="right")
    width = hi - lo
    idx = lo[:, None] + rng.integers(0, np.maximum(width, 1)[:, None], size=(len(mass), k))
    narrow = width <= 1
    if narrow.any():
        near = pos_idx[narrow, None] + rng.integers(-25, 26, size=(int(narrow.sum()), k))
        idx[narrow] = near
    return np.clip(idx, 0, len(pool_mass) - 1)


@torch.no_grad()
def embed_pool(
    model: ContrastiveNet, pool_fp_packed: np.ndarray, device, batch: int = 8192, pool_bits: torch.Tensor | None = None
) -> np.ndarray:
    """Unit-norm molecule embeddings ``(n_pool, d)`` float16 for every pool structure.

    ``pool_bits`` is the optional device-resident uint8 bit matrix from :func:`pool_bits_on_device`.
    """
    model.eval()
    out = []
    for i in range(0, len(pool_fp_packed), batch):
        if pool_bits is not None:
            bits = pool_bits[i : i + batch].float()
        else:
            bits = torch.from_numpy(np.unpackbits(pool_fp_packed[i : i + batch], axis=1).astype(np.float32)).to(device)
        out.append(model.encode_molecule(bits).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def pool_bits_on_device(pool_fp_packed: np.ndarray, device, chunk: int = 50_000) -> torch.Tensor:
    """All pool fingerprints as a uint8 ``(n_pool, n_bits)`` tensor on ``device`` (unpacked in chunks)."""
    parts = [
        torch.from_numpy(np.unpackbits(pool_fp_packed[i : i + chunk], axis=1)).to(device)
        for i in range(0, len(pool_fp_packed), chunk)
    ]
    return torch.cat(parts)


@torch.no_grad()
def embed_spectra(model: ContrastiveNet, X: np.ndarray, meta: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        z = model.encode_spectrum(
            torch.from_numpy(X[i : i + batch].astype(np.float32)).to(device),
            torch.from_numpy(meta[i : i + batch]).to(device),
        )
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def aggregate_embeddings(molecule_ids, emb: np.ndarray) -> tuple[list, np.ndarray]:
    """One unit vector per molecule: the renormalised mean of its spectra's embeddings."""
    codes, uniques = pd.factorize(pd.Series(list(molecule_ids)))
    total = np.zeros((len(uniques), emb.shape[1]), dtype=np.float32)
    np.add.at(total, codes, emb.astype(np.float32))
    return list(uniques), total / (np.linalg.norm(total, axis=1, keepdims=True) + 1e-9)


def rank_contrastive(
    cp,
    pool: pd.DataFrame,
    pool_emb,
    mol_ids,
    mol_emb,
    masses,
    ppm: float,
    prior: np.ndarray | None = None,
    prior_weight: float = 0.0,
    scale=1.0,
    prior_topk: int | None = None,
    consensus: dict | None = None,
    retrieval_weight: float = 0.0,
    pool_fp_packed: np.ndarray | None = None,
) -> dict:
    """``{molecule_id: [pool index, ...]}`` best first; lists are padded to 25 like the team's ranking.

    **Ensemble:** pass lists for ``pool_emb``, ``mol_emb`` and ``scale`` (one entry per model, same molecule order) and
    each candidate's logit is the average of ``scale_i * cosine_i`` over the models. With a single model the logit is
    ``scale * cosine``; ordering by it equals ordering by cosine.

    With ``prior`` (one value per pool structure) ``prior_weight * prior`` is added to the logit. With ``prior_topk``
    the prior only reorders the ``prior_topk`` best candidates (so it cannot change which candidates are kept).

    With ``consensus`` (``{molecule: fingerprint}`` from analog-spectrum retrieval), ``retrieval_weight`` times the
    Tanimoto similarity between each candidate and the molecule's consensus fingerprint is added to the logit first;
    ``pool_fp_packed`` supplies the candidates' bits. The prior, if any, is applied on top of that blended score.
    """
    ensemble = isinstance(pool_emb, list | tuple)
    pool_embs = list(pool_emb) if ensemble else [pool_emb]
    mol_embs = list(mol_emb) if ensemble else [mol_emb]
    scales = list(scale) if ensemble else [scale]
    pool_mass = pool.mass.to_numpy()
    rankings = {}
    for m, mol in enumerate(mol_ids):
        mass = float(masses[mol])
        idx = cp.window_indices(pool_mass, mass, ppm)
        if len(idx):
            parts = zip(pool_embs, mol_embs, scales, strict=True)
            logit = sum(s * (pe[idx].astype(np.float32) @ me[m]) for pe, me, s in parts) / len(pool_embs)
            if consensus is not None and retrieval_weight and mol in consensus:
                logit = logit + retrieval_weight * F.soft_tanimoto(
                    consensus[mol], np.unpackbits(pool_fp_packed[idx], axis=1)
                )
            order = np.argsort(-logit, kind="stable")
            if prior is not None and prior_weight:
                head = order if prior_topk is None else order[:prior_topk]
                adjusted = logit[head] + prior_weight * prior[idx[head]]
                order = np.concatenate([head[np.argsort(-adjusted, kind="stable")], order[len(head) :]])
            idx = idx[order]
        rankings[mol] = cp.fill_to_k(idx, pool_mass, mass)
    return rankings


def build_model(cfg: dict, n_bits: int) -> ContrastiveNet:
    """Network described by a run config (``data`` spectrum settings + ``model`` sizes)."""
    return ContrastiveNet(n_bins=F.input_dim(**F.spec_args(cfg["data"])), n_bits=n_bits, **cfg["model"])


def load_checkpoints(paths, n_bits: int, device, data_cfg: dict | None = None) -> list[ContrastiveNet]:
    """Load saved models (``{"state", "cfg"}`` checkpoints) in eval mode, each built from its own saved config.

    With ``data_cfg`` every checkpoint must use the same spectrum representation (bin width, neutral-loss block),
    because an ensemble embeds the same spectrum arrays with all models.
    """
    models = []
    for path in paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if data_cfg is not None and F.spec_args(ckpt["cfg"]["data"]) != F.spec_args(data_cfg):
            got, want = F.spec_args(ckpt["cfg"]["data"]), F.spec_args(data_cfg)
            raise ValueError(f"{path}: spectrum settings {got} differ from {want}")
        model = build_model(ckpt["cfg"], n_bits).to(device)
        model.load_state_dict(ckpt["state"])
        models.append(model.eval())
    return models
