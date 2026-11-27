"""Architecture 3: contrastive spectrum-molecule retrieval with a peak transformer. Owner: Member 7.

Why: after reranking, 70-80% of the validation MRR we still lose is the answer beaten by a same-formula
isomer. The first stage predicts a 2,048-bit fingerprint and compares it with Tanimoto, and close isomers share
almost every bit, so that comparison cannot separate them however good the encoder is. This model is trained to
do exactly that job: pick the right structure out of the candidates that share its mass.

    spectrum peaks --PeakEncoder (transformer)--> h_i  (one per spectrum)
    {h_1..h_n}     --GatedAttentionPool--------> q    (one per molecule; starts as the mean, learns weights)
    candidate      --MoleculeEncoder (MLP on Morgan bits)--> m
    score(q, m) = cos(q, m) * scale                        (CLIP-style, learned temperature)

Training (InfoNCE): for each training molecule the positive is its own structure; the negatives are the other
molecules in the batch plus ``n_neg`` structures drawn from its own +-ppm mass window of the candidate pool, which
are mostly same-formula isomers (hard negatives). A small fingerprint head on q (BCE) keeps the embedding
chemically grounded early in training.

Peaks are tokens (exact m/z and neutral loss as Fourier features, plus intensity), so nothing is binned. The pool
uses Ilse et al.'s gated attention with the scoring layer initialised to zero: it is exactly mean pooling at the
start and only moves away from it if that helps the contrastive loss, which is the fair test of whether learned
pooling beats the mean when the encoder is trained with it (Member 7's Arch 1 / 2 only had a frozen encoder).
"""

from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from common.sets import pad_sets, sample_sets

N_ADDUCTS_DEFAULT = 16


# ---------- data ----------


def top_peaks(mz: np.ndarray, intensity: np.ndarray, n: int = 64,
              chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ``n`` most intense peaks of each padded row (padding = intensity 0), sorted by m/z."""
    if mz.shape[1] <= n:
        return mz.astype(np.float32), intensity.astype(np.float32)
    out_mz = np.zeros((len(mz), n), np.float32)
    out_in = np.zeros((len(mz), n), np.float32)
    for s in range(0, len(mz), chunk):  # chunks: 1.7M training spectra would need ~4 GB of temporaries at once
        m, it = mz[s:s + chunk], intensity[s:s + chunk]
        keep = np.argpartition(-it, n - 1, axis=1)[:, :n]
        mz_k, in_k = np.take_along_axis(m, keep, 1), np.take_along_axis(it, keep, 1)
        order = np.argsort(np.where(in_k > 0, mz_k, np.inf), axis=1, kind="stable")
        mz_k, in_k = np.take_along_axis(mz_k, order, 1), np.take_along_axis(in_k, order, 1)
        mz_k[in_k <= 0] = 0.0
        out_mz[s:s + chunk], out_in[s:s + chunk] = mz_k, in_k
    return out_mz, out_in


def window_bounds(pool_mass: np.ndarray, masses: np.ndarray, ppm: float) -> tuple[np.ndarray, np.ndarray]:
    """[lo, hi) of the +-ppm window of each mass in a mass-sorted pool."""
    masses = np.asarray(masses, dtype=np.float64)
    tol = masses * ppm * 1e-6
    return (np.searchsorted(pool_mass, masses - tol, side="left"),
            np.searchsorted(pool_mass, masses + tol, side="right"))


def sample_negatives(rng: np.random.Generator, pos: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_neg: int,
                     n_pool: int) -> np.ndarray:
    """(B, n_neg) pool indices from each positive's own mass window, never the positive itself.

    A window with no other structure falls back to random pool structures (easy negatives).
    """
    pos, lo, hi = (np.asarray(a, dtype=np.int64) for a in (pos, lo, hi))
    width = hi - lo  # includes the positive
    others = np.maximum(width - 1, 0)
    u = rng.random((len(pos), n_neg))
    r = lo[:, None] + np.floor(u * np.maximum(others, 1)[:, None]).astype(np.int64)
    r = np.where(r >= pos[:, None], r + 1, r)  # skip the positive
    rand = rng.integers(0, n_pool, size=r.shape)
    rand = np.where(rand == pos[:, None], (rand + 1) % n_pool, rand)
    return np.where((others > 0)[:, None], r, rand)


# ---------- model ----------


class PeakEncoder(nn.Module):
    """One spectrum -> one vector. Tokens: a summary token (precursor + adduct) and one token per peak."""

    def __init__(self, d: int = 256, layers: int = 4, heads: int = 8, dropout: float = 0.1, n_freq: int = 32,
                 n_adducts: int = N_ADDUCTS_DEFAULT):
        super().__init__()
        # wavelengths from 0.01 Da (separates fragments of nearly equal mass) to 1,000 Da (coarse position)
        self.register_buffer("freqs", 2 * math.pi / torch.logspace(-2.0, 3.0, n_freq))
        self.peak = nn.Sequential(nn.Linear(4 * n_freq + 2, d), nn.GELU(), nn.Linear(d, d))
        self.summary = nn.Sequential(nn.Linear(2 * n_freq, d), nn.GELU(), nn.Linear(d, d))
        self.adduct = nn.Embedding(n_adducts, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def fourier(self, value: torch.Tensor) -> torch.Tensor:
        angle = value.unsqueeze(-1) * self.freqs
        return torch.cat([angle.sin(), angle.cos()], dim=-1)

    def forward(self, mz, intensity, precursor, adduct):
        loss = precursor.unsqueeze(1) - mz  # neutral loss of each peak
        rel = intensity / intensity.amax(1, keepdim=True).clamp(min=1e-6)
        tok = self.peak(torch.cat([self.fourier(mz), self.fourier(loss), intensity.unsqueeze(-1),
                                   rel.unsqueeze(-1)], dim=-1))
        summ = (self.summary(self.fourier(precursor)) + self.adduct(adduct.clamp(0, self.adduct.num_embeddings - 1)))
        pad = torch.cat([torch.zeros_like(intensity[:, :1], dtype=torch.bool), intensity <= 0], dim=1)
        out = self.encoder(torch.cat([summ.unsqueeze(1), tok], dim=1), src_key_padding_mask=pad)
        return self.norm(out[:, 0])


class GatedAttentionPool(nn.Module):
    """Gated attention over a set (Ilse et al., 2018); identical to the mean at initialisation."""

    def __init__(self, d: int, att: int = 128):
        super().__init__()
        self.V, self.U, self.w = nn.Linear(d, att), nn.Linear(d, att), nn.Linear(att, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, h: torch.Tensor, mask: torch.Tensor, mode: str = "attention"):
        if mode == "mean":
            m = mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1), None
        s = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1).float()
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)
        return (a.unsqueeze(-1).to(h.dtype) * h).sum(1), a


class MoleculeEncoder(nn.Module):
    """Candidate structure (Morgan bits) -> vector."""

    def __init__(self, fp_bits: int = 2048, hidden: int = 1024, d: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fp_bits, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d))

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        return self.net(bits)


class Retriever(nn.Module):
    def __init__(self, d: int = 256, layers: int = 4, heads: int = 8, emb: int = 256, fp_bits: int = 2048,
                 n_adducts: int = N_ADDUCTS_DEFAULT, dropout: float = 0.1):
        super().__init__()
        self.spectra = PeakEncoder(d, layers, heads, dropout, n_adducts=n_adducts)
        self.pool = GatedAttentionPool(d)
        self.mol = MoleculeEncoder(fp_bits, 4 * emb, d, dropout)
        self.q_proj, self.m_proj = nn.Linear(d, emb), nn.Linear(d, emb)
        self.fp_head = nn.Linear(d, fp_bits)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def scale(self) -> torch.Tensor:
        return self.log_scale.clamp(max=math.log(100.0)).exp()

    def query(self, h: torch.Tensor, mask: torch.Tensor, mode: str = "attention"):
        pooled, att = self.pool(h, mask, mode)
        return F.normalize(self.q_proj(pooled).float(), dim=-1), self.fp_head(pooled), att

    def molecules(self, bits: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.m_proj(self.mol(bits)).float(), dim=-1)


def info_nce(q: torch.Tensor, m_pos: torch.Tensor, m_neg: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """q, m_pos: (B, E); m_neg: (B, H, E). Each anchor must pick its own positive over the batch's other
    positives and its own hard negatives."""
    logits = torch.cat([q @ m_pos.T, torch.einsum("be,bhe->bh", q, m_neg)], dim=1) * scale
    return F.cross_entropy(logits, torch.arange(len(q), device=q.device))


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> shifts) & 1).flatten(-2).float()


# ---------- training ----------


def train_retriever(model, spec, groups, train_keys, key_index, pool_fp, pool_mass, device, *, epochs=15,
                    batch=256, n_neg=31, lr=3e-4, max_set=8, ppm=10.0, aux_weight=0.3, max_minutes=100.0,
                    n_gpu=1, seed=0, keep_prob=None, score_fn=None, log=print):
    """Train ``model`` with InfoNCE on training molecules; keep the epoch with the best ``score_fn(model)``.

    spec: dict of per-row training spectra arrays mz, intensity (N, P), precursor (N,), adduct (N,).
    groups: {key: row indices into spec}; pool_mass must be sorted (Member 3's pool is).
    keep_prob: optional (len(train_keys),) chance that a molecule takes part in an epoch (re-drawn every epoch),
    e.g. < 1 for the synthetic library so natural products weigh more.
    The learning rate warms up for 5%, then follows a cosine to zero over training progress, measured as the larger
    of steps done / steps planned and minutes used / ``max_minutes``: a run cut by the time budget still anneals.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    spec_net = nn.DataParallel(model.spectra, device_ids=list(range(n_gpu))) if n_gpu > 1 else model.spectra
    fp_dev = torch.as_tensor(pool_fp, device=device)
    t = {k: torch.as_tensor(v) for k, v in spec.items()}
    t["precursor"] = t["precursor"].float()
    keys = list(train_keys)
    pos_all = np.array([key_index[k] for k in keys], dtype=np.int64)
    lo_all, hi_all = window_bounds(pool_mass, pool_mass[pos_all], ppm)
    keep_prob = np.ones(len(keys)) if keep_prob is None else np.asarray(keep_prob, dtype=float)
    steps_total = max(int(epochs * keep_prob.sum() // batch), 2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    step = 0

    def progress():
        return max(step / steps_total, (time.time() - start) / 60 / max_minutes)

    def set_lr():
        p = progress()
        f = p / 0.05 if p < 0.05 else 0.5 * (1 + math.cos(math.pi * min((p - 0.05) / 0.95, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr * max(f, 0.0)
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else contextlib.nullcontext
    start, best, best_state, hist = time.time(), -math.inf, None, []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(np.flatnonzero(rng.random(len(keys)) < keep_prob))
        tot = {"nce": 0.0, "bce": 0.0, "n": 0}
        for s in range(0, len(order) - batch + 1, batch):
            sel = order[s:s + batch]
            sets = sample_sets(groups, [keys[j] for j in sel], rng, max_set=max_set)
            rows = np.concatenate(sets)
            local = np.split(np.arange(len(rows)), np.cumsum([len(x) for x in sets])[:-1])
            neg = sample_negatives(rng, pos_all[sel], lo_all[sel], hi_all[sel], n_neg, len(pool_mass))
            r = torch.as_tensor(rows)
            with autocast():
                h = spec_net(t["mz"][r].to(device), t["intensity"][r].to(device), t["precursor"][r].to(device),
                             t["adduct"][r].to(device))
                hs, mask = pad_sets(local, h)
                q, fp_logits, _ = model.query(hs, mask)
                idx = torch.as_tensor(np.concatenate([pos_all[sel][:, None], neg], 1), device=device)
                bits = _unpack(fp_dev[idx.flatten()])
                m = model.molecules(bits).view(len(sel), 1 + n_neg, -1)
                nce = info_nce(q, m[:, 0], m[:, 1:], model.scale())
                bce = F.binary_cross_entropy_with_logits(fp_logits.float(), bits.view(len(sel), 1 + n_neg, -1)[:, 0])
                loss = nce + aux_weight * bce
            set_lr()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            tot["nce"] += float(nce.detach())
            tot["bce"] += float(bce.detach())
            tot["n"] += 1
            if progress() >= 1.0:
                break
        row = {"epoch": ep + 1, "minutes": round((time.time() - start) / 60, 2),
               "train_nce": tot["nce"] / max(tot["n"], 1), "train_bce": tot["bce"] / max(tot["n"], 1),
               "scale": float(model.scale().detach())}
        if score_fn is not None:
            model.eval()
            row["val_mrr"] = float(score_fn(model))
        hist.append(row)
        log(row)
        crit = row.get("val_mrr", -row["train_nce"])
        if crit > best:
            best, best_state = crit, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress() >= 1.0:
            if ep + 1 < epochs:
                log(f"retriever: schedule finished after {ep + 1} epochs (time budget)")
            break
    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(hist)


# ---------- inference ----------


@torch.no_grad()
def spectrum_vectors(model, mz, intensity, precursor, adduct, device, batch=2048):
    out = []
    for i in range(0, len(mz), batch):
        sl = slice(i, i + batch)
        out.append(model.spectra(torch.as_tensor(mz[sl], device=device),
                                 torch.as_tensor(intensity[sl], device=device),
                                 torch.as_tensor(precursor[sl], dtype=torch.float32, device=device),
                                 torch.as_tensor(adduct[sl], device=device)).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, model.spectra.norm.normalized_shape[0])


@torch.no_grad()
def query_vectors(model, sets, H, device, mode="attention", chunk=1024):
    """(n_sets, E) numpy, one normalised query vector per molecule, plus the attention weights."""
    out, atts = [], []
    for s in range(0, len(sets), chunk):
        h, mask = pad_sets(sets[s:s + chunk], H)
        q, _, a = model.query(h.to(device), mask.to(device), mode)
        out.append(q.cpu().numpy())
        if a is not None:
            atts += [a[i, :len(x)].cpu().numpy() for i, x in enumerate(sets[s:s + chunk])]
    return np.concatenate(out), atts


@torch.no_grad()
def molecule_vectors(model, pool_fp, indices, device, chunk=8192):
    """{pool index: vector} for the given indices (deduplicated)."""
    uniq = np.unique(np.asarray(indices, dtype=np.int64))
    vecs = []
    for s in range(0, len(uniq), chunk):
        bits = _unpack(torch.as_tensor(pool_fp[uniq[s:s + chunk]], device=device))
        vecs.append(model.molecules(bits).cpu().numpy())
    mat = np.concatenate(vecs) if vecs else np.zeros((0, model.m_proj.out_features), np.float32)
    return uniq, mat


def window_scores(model, pool_fp, pool_mass, masses, q, device, ppm=10.0):
    """{molecule: (window indices, cosine score)} in the same form as ``full.window_similarity``."""
    lo, hi = window_bounds(pool_mass, masses.to_numpy(), ppm)
    uniq, mat = molecule_vectors(model, pool_fp, np.concatenate([np.arange(a, b) for a, b in zip(lo, hi,
                                                                                               strict=True)]
                                                                 + [np.zeros(0, np.int64)]), device)
    out = {}
    for (mol, _), a, b, qi in zip(masses.items(), lo, hi, q, strict=True):
        idx = np.arange(a, b)
        rows = np.searchsorted(uniq, idx)
        out[mol] = (idx, (mat[rows] @ qi).astype(np.float32) if len(idx) else np.zeros(0, np.float32))
    return out


def fuse_windows(a: dict, b: dict, wb: float = 1.0) -> dict:
    """Per-window z(a) + wb * z(b) of two score dicts over the same windows."""
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-6) if len(x) else x
    return {m: (a[m][0], z(a[m][1]) + wb * z(b[m][1])) for m in a}
