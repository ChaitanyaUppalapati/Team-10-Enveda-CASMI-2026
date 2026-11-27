"""S3: spectrum -> Morgan fingerprint predictor (residual MLP) and fingerprint-likelihood ranking.

Input features per spectrum (built on the fly from the 0.01 Da library CSR):
- fragment peaks binned at 0.1 Da up to 1000 m/z (10,000 features)
- neutral losses (precursor - fragment) binned at 0.1 Da up to 500 Da (5,000 features)
- precursor mass, ionisation mode, adduct one-hot, collision energy (+ missing flag), timsTOF flag
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from .chem import TEST_ADDUCTS
from .fingerprints import FP_BITS
from .spectra import BIN

COARSE = 0.1
N_FRAG = int(1000 / COARSE)
N_NL = int(500 / COARSE)
N_META = 1 + 1 + len(TEST_ADDUCTS) + 2 + 1
N_IN = N_FRAG + N_NL + N_META
ADDUCT_INDEX = {a: i for i, a in enumerate(TEST_ADDUCTS)}


def meta_features(precursor, mode, adduct, ce, timstof) -> np.ndarray:
    n = len(precursor)
    f = np.zeros((n, N_META), dtype=np.float32)
    f[:, 0] = np.asarray(precursor) / 1000.0
    f[:, 1] = np.asarray(mode, dtype=np.float32)
    ai = np.array([ADDUCT_INDEX.get(a, -1) for a in adduct])
    ok = ai >= 0
    f[np.flatnonzero(ok), 2 + ai[ok]] = 1.0
    ce = np.asarray(ce, dtype=np.float64)
    f[:, 2 + len(TEST_ADDUCTS)] = np.nan_to_num(ce / 100.0, nan=0.0)
    f[:, 3 + len(TEST_ADDUCTS)] = np.isnan(ce)
    f[:, 4 + len(TEST_ADDUCTS)] = np.asarray(timstof, dtype=np.float32)
    return f


def dense_batch(X: sp.csr_matrix, rows: np.ndarray, precursor: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Coarse fragment + neutral-loss bins for ``rows`` of the fine CSR, plus meta features."""
    sub = X[rows].tocoo()
    out = np.zeros((len(rows), N_IN), dtype=np.float32)
    mz = sub.col * BIN
    fb = (mz / COARSE).astype(np.int64)
    ok = fb < N_FRAG
    np.add.at(out, (sub.row[ok], fb[ok]), sub.data[ok])
    nl = precursor[sub.row] - mz
    nb = (nl / COARSE).astype(np.int64)
    ok = (nl > 0.5) & (nb < N_NL)
    np.add.at(out, (sub.row[ok], N_FRAG + nb[ok]), sub.data[ok])
    out[:, N_FRAG + N_NL:] = meta
    return out


def batch_tensor(X: sp.csr_matrix, rows: np.ndarray, precursor: np.ndarray, meta: np.ndarray,
                 device) -> torch.Tensor:
    """Same features as ``dense_batch`` but scattered on ``device`` (much faster for training)."""
    sub = X[rows].tocoo()
    mz = sub.col * BIN
    fb = (mz / COARSE).astype(np.int64)
    nl = precursor[sub.row] - mz
    nb = (nl / COARSE).astype(np.int64)
    okf = fb < N_FRAG
    okn = (nl > 0.5) & (nb < N_NL)
    r = np.concatenate([sub.row[okf], sub.row[okn]])
    c = np.concatenate([fb[okf], N_FRAG + nb[okn]])
    v = np.concatenate([sub.data[okf], sub.data[okn]]).astype(np.float32)
    out = torch.zeros((len(rows), N_IN), device=device)
    out.index_put_((torch.from_numpy(r).to(device), torch.from_numpy(c).to(device)),
                   torch.from_numpy(v).to(device), accumulate=True)
    out[:, N_FRAG + N_NL:] = torch.from_numpy(meta).to(device)
    return out


class Block(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width, width))

    def forward(self, x):
        return x + self.net(x)


class FPNet(nn.Module):
    def __init__(self, n_in=N_IN, width=2048, depth=3, dropout=0.2, n_out=FP_BITS):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[Block(width, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_out))

    def embed(self, x):
        return self.blocks(self.inp(x))

    def forward(self, x):
        return self.out(self.embed(x))


def fp_loglik_scores(logits_mol: np.ndarray, cand_bits: np.ndarray) -> np.ndarray:
    """Bernoulli log-likelihood of each candidate fingerprint under predicted bit probabilities.

    logits_mol: (FP_BITS,) pooled logits for one molecule; cand_bits: (n_cand, FP_BITS) 0/1.
    """
    lp = -np.logaddexp(0, -logits_mol)   # log sigmoid
    lq = -np.logaddexp(0, logits_mol)    # log (1 - sigmoid)
    return cand_bits @ (lp - lq) + lq.sum()


def fp_cosine_scores(logits_mol: np.ndarray, cand_bits: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-logits_mol))
    num = cand_bits @ p
    den = np.sqrt(cand_bits.sum(1)) * np.linalg.norm(p) + 1e-9
    return num / den
