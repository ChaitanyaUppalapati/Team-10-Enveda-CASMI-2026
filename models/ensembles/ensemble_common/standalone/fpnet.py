"""Spectrum -> Morgan fingerprint MLP (PyTorch, BCE). Trains on the Kaggle GPU; not run locally."""

from __future__ import annotations

import numpy as np


class TorchFingerprintModel:
    """``fit(X, Y)`` then ``predict(X)`` -> per-bit probabilities (float32)."""

    def __init__(self, hidden: int = 1024, dropout: float = 0.2, epochs: int = 12, batch_size: int = 512,
                 lr: float = 1e-3, weight_decay: float = 1e-5, seed: int = 2026, device: str | None = None):
        self.hidden, self.dropout, self.epochs = hidden, dropout, epochs
        self.batch_size, self.lr, self.weight_decay, self.seed = batch_size, lr, weight_decay, seed
        self.device = device
        self.net = None

    def _build(self, d_in: int, d_out: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(d_in, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, d_out),
        )

    def fit(self, X: np.ndarray, Y: np.ndarray, log=print) -> TorchFingerprintModel:
        import torch

        torch.manual_seed(self.seed)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.net = self._build(X.shape[1], Y.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        n = len(X)
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen).numpy()
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = torch.from_numpy(X[idx].astype(np.float32)).to(dev)
                yb = torch.from_numpy(Y[idx].astype(np.float32)).to(dev)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            log(f"epoch {epoch + 1}/{self.epochs} bce={total / n:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.from_numpy(X[i : i + 2048].astype(np.float32)).to(self.device)
                out.append(torch.sigmoid(self.net(xb)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class PackedFingerprintMLP:
    """Binned spectrum + side features -> Morgan bit probabilities (BCE), for the team-split notebook.

    ``fit(X, S, target_idx, fp_packed)``: X (n, bins) float16 spectra, S (n, d) side features,
    target_idx (n,) rows of ``fp_packed`` (pool fingerprints, packed uint8). Targets are unpacked on
    the GPU per batch, and X is kept on the GPU when it fits, so ~1M training spectra train quickly.
    Fixed epochs with a one-cycle schedule; validation BCE is only logged, never used to pick a
    checkpoint, so the validation MRR stays honest.
    """

    def __init__(self, hidden: int = 2048, depth: int = 3, dropout: float = 0.2, epochs: int = 20,
                 batch_size: int = 1024, lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 2026,
                 max_minutes: float = 150.0, device: str | None = None):
        self.hidden, self.depth, self.dropout, self.epochs = hidden, depth, dropout, epochs
        self.batch_size, self.lr, self.weight_decay, self.seed = batch_size, lr, weight_decay, seed
        self.max_minutes, self.device = max_minutes, device
        self.net, self.history = None, []

    def _build(self, d_in: int, d_out: int):
        import torch.nn as nn

        layers, d = [], d_in
        for _ in range(self.depth):
            layers += [nn.Linear(d, self.hidden), nn.LayerNorm(self.hidden), nn.GELU(), nn.Dropout(self.dropout)]
            d = self.hidden
        layers.append(nn.Linear(d, d_out))
        return nn.Sequential(*layers)

    def fit(self, X, S, target_idx, fp_packed, val=None, log=print):
        import contextlib
        import math
        import time

        import torch

        torch.manual_seed(self.seed)
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = str(dev)
        n, n_bits = len(X), fp_packed.shape[1] * 8
        self._fp = fp_packed
        self.net =self._build(X.shape[1] + S.shape[1], n_bits).to(dev)
        FP = torch.from_numpy(np.ascontiguousarray(fp_packed)).to(dev)
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float16))
        St = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float16))
        Tt = torch.from_numpy(np.array(target_idx, dtype=np.int64))
        store = dev
        try:
            Xt, St, Tt = Xt.to(dev), St.to(dev), Tt.to(dev)
        except RuntimeError:  # not enough GPU memory: keep the data on the CPU and copy per batch
            store = torch.device("cpu")
            Xt, St, Tt = Xt.cpu(), St.cpu(), Tt.cpu()
            torch.cuda.empty_cache()
        log(f"fingerprint MLP: {n:,} spectra, data on {store}, model on {dev}")
        cuda = dev.type == "cuda"
        shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)
        steps = max(self.epochs * math.ceil(n / self.batch_size), 2)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
        autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if cuda else contextlib.nullcontext
        loss_fn = torch.nn.BCEWithLogitsLoss()
        start, done = time.time(), 0
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=store)
            total, count = torch.zeros((), device=dev), 0
            for i in range(0, n, self.batch_size):
                b = perm[i : i + self.batch_size]
                xb = torch.cat([Xt[b], St[b]], dim=1).to(dev, non_blocking=True).float()
                yb = ((FP[Tt[b].to(dev)].unsqueeze(-1) >> shifts) & 1).flatten(1).float()
                with autocast():
                    loss = loss_fn(self.net(xb), yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if done < steps - 1:
                    sched.step()
                done += 1
                total, count = total + loss.detach(), count + 1
            row = {"epoch": epoch + 1, "minutes": round((time.time() - start) / 60, 2),
                   "train_bce": round(total.item() / max(count, 1), 5)}
            if val is not None:
                row["val_bce"] = round(self.bce(*val), 5)
            self.history.append(row)
            log(f"  {row}")
            if (time.time() - start) / 60 > self.max_minutes:
                log("  stopped at the time budget")
                break
        return self

    def _logits(self, X, S, batch: int = 4096):
        import torch

        self.net.eval()
        dev = torch.device(self.device)
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = np.hstack([np.asarray(X[i : i + batch], np.float32), np.asarray(S[i : i + batch], np.float32)])
                yield i, self.net(torch.from_numpy(xb).to(dev)).float()

    def bce(self, X, S, target_idx, fp_packed=None) -> float:
        """Mean BCE against the packed fingerprints of ``target_idx`` (``fp_packed`` defaults to training's)."""
        import torch

        fp_packed = self._fp if fp_packed is None else fp_packed
        total, n = 0.0, 0
        for i, logits in self._logits(X, S):
            bits = np.unpackbits(fp_packed[np.asarray(target_idx[i : i + len(logits)])], axis=1)
            y = torch.from_numpy(bits[:, : logits.shape[1]].astype(np.float32)).to(logits.device)
            total += torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
            n += y.numel()
        return total / max(n, 1)

    def predict(self, X, S) -> np.ndarray:
        import torch

        out = np.zeros((len(X), self.net[-1].out_features), dtype=np.float32)
        for i, logits in self._logits(X, S):
            out[i : i + len(logits)] = torch.sigmoid(logits).cpu().numpy()
        return out
