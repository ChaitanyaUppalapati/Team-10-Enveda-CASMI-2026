"""Dual-branch spectrum + precursor network. Owner: Member 4."""

from __future__ import annotations

import torch
from torch import nn

from . import features as F


class DualBranchNet(nn.Module):
    """Spectrum branch + precursor/adduct/collision-energy branch -> fingerprint logits + element counts.

    The formula of an unknown molecule is not available at test time, so it is an auxiliary target
    (log element counts); at scoring time the predicted counts are compared with each candidate's formula.
    """

    def __init__(
        self,
        n_bins: int = F.N_BINS,
        n_meta: int = F.N_META,
        n_bits: int = 2048,
        n_elements: int = len(F.ELEMENTS),
        spec_dim: int = 512,
        meta_dim: int = 64,
        hidden: int = 1024,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.spectrum = nn.Sequential(
            nn.Linear(n_bins, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, spec_dim)
        )
        self.meta = nn.Sequential(nn.Linear(n_meta, meta_dim), nn.GELU(), nn.Linear(meta_dim, meta_dim))
        self.trunk = nn.Sequential(
            nn.GELU(), nn.Linear(spec_dim + meta_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.fingerprint = nn.Linear(hidden, n_bits)
        self.formula = nn.Linear(hidden, n_elements)

    def forward(self, spectrum: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([self.spectrum(spectrum), self.meta(meta)], dim=-1))
        return self.fingerprint(h), self.formula(h)
