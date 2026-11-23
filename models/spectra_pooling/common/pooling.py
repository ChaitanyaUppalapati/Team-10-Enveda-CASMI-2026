"""Spectrum encoder and the three ways to pool a molecule's spectra. Owner: Member 7.

Every query (molecule) has 1-16 spectra. Each spectrum is encoded on its own into an
embedding ``h`` and a per-spectrum fingerprint prediction. The pooling heads turn the
variable-size set {h_1..h_n} into ONE fingerprint prediction for the molecule:

- ``mean_probabilities``: no training, average the per-spectrum fingerprint probabilities
  (what Member 2's CNN scorer does; our baseline to beat).
- ``DeepSetsPool`` (arch2): rho(mean_i phi(h_i)). Order-free, any set size, every
  spectrum counts equally.
- ``AttentionPool`` (arch1): gated attention (Ilse et al., 2018). Learns a weight per
  spectrum, so noisy or uninformative spectra can be down-weighted.

Sets are padded to the largest set in a batch; ``mask`` is True for real spectra.
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(sizes: list[int], dropout: float = 0.0, last_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2 or last_act:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class SpectrumEncoder(nn.Module):
    """Binned spectrum (+ metadata) -> embedding h, plus a per-spectrum fingerprint head."""

    def __init__(self, n_bins: int, n_meta: int, fp_size: int, hidden: int = 1024, emb: int = 512,
                 dropout: float = 0.2):
        super().__init__()
        self.body = mlp([n_bins + n_meta, hidden, emb], dropout)
        self.head = nn.Linear(emb, fp_size)

    def embed(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([x, meta], dim=-1))

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x, meta))


def mean_probabilities(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average per-spectrum probabilities over the real spectra of each set.

    probs: (B, N, F), mask: (B, N) bool -> (B, F)
    """
    m = mask.unsqueeze(-1).to(probs.dtype)
    return (probs * m).sum(1) / m.sum(1).clamp(min=1)


class DeepSetsPool(nn.Module):
    """rho( mean_i phi(h_i) ), with log(set size) appended so rho knows how much evidence it has."""

    def __init__(self, emb: int, fp_size: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.phi = mlp([emb, hidden, hidden], dropout)
        self.rho = mlp([hidden + 1, hidden, fp_size], dropout, last_act=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        m = mask.unsqueeze(-1).to(h.dtype)
        n = m.sum(1).clamp(min=1)
        pooled = (self.phi(h) * m).sum(1) / n
        return self.rho(torch.cat([pooled, torch.log(n)], dim=-1)), None


class AttentionPool(nn.Module):
    """Gated attention pooling: weights a_i = softmax_i( w . (tanh(V h_i) * sigmoid(U h_i)) ).

    Returns logits (B, F) and the attention weights (B, N), which sum to 1 over real spectra.
    """

    def __init__(self, emb: int, fp_size: int, hidden: int = 512, att: int = 128, dropout: float = 0.2):
        super().__init__()
        self.phi = mlp([emb, hidden, hidden], dropout)
        self.V = nn.Linear(hidden, att)
        self.U = nn.Linear(hidden, att)
        self.w = nn.Linear(att, 1)
        self.rho = mlp([hidden + 1, hidden, fp_size], dropout, last_act=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        z = self.phi(h)
        scores = self.w(torch.tanh(self.V(z)) * torch.sigmoid(self.U(z))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * z).sum(1)
        n = mask.sum(1, keepdim=True).to(h.dtype).clamp(min=1)
        return self.rho(torch.cat([pooled, torch.log(n)], dim=-1)), weights


def build_pool(name: str, emb: int, fp_size: int, **kw) -> nn.Module:
    pools = {"deepsets": DeepSetsPool, "attention": AttentionPool}
    if name not in pools:
        raise ValueError(f"Unknown pooling {name!r}; choose from {sorted(pools)}")
    return pools[name](emb, fp_size, **kw)
