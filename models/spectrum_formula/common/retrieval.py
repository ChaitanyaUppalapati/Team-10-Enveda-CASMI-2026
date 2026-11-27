"""Retrieval of analog spectra: nearest library spectra vote with their structures. Owner: Member 4.

Related compounds (same natural-product family) tend to give similar spectra, so even when the exact molecule has no
library spectrum, close relatives often do. Library spectra are embedded with a trained spectrum encoder; for each
query molecule the nearest library spectra are found by cosine similarity, their structures' fingerprints are
averaged (weighted by similarity) into a *consensus fingerprint*, and each candidate is scored by its Tanimoto
similarity to that consensus. The retrieval score is blended with the model's own score at ranking time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import contrastive as C
from . import features as F
from . import team


def library_rows(proto: team.Protocol, data_cfg: dict) -> pd.DataFrame:
    """Library spectra: the team's training rows (held-out structures' spectra are already excluded)."""
    rows = proto.cp.select_train_rows(proto.meta, proto.lib_mask, per_structure=data_cfg.get("per_structure", 4))
    return rows[rows.inchikey14.astype(str).isin(proto.key_index.index)].reset_index(drop=True)


def embed_library(
    model, proto: team.Protocol, rows: pd.DataFrame, spec_args: dict, device, chunk: int = 40_000
) -> np.ndarray:
    """Spectrum embeddings ``(n, d)`` float16 for ``rows`` (sorted by ``row_id``), read and embedded chunk by chunk."""
    parts = []
    for i in range(0, len(rows), chunk):
        X, meta = team.spectra_arrays(proto.train_path, rows.iloc[i : i + chunk], **spec_args)
        parts.append(C.embed_spectra(model, X, meta, device).astype(np.float16))
    return np.concatenate(parts)


@torch.no_grad()
def knn(query_emb: np.ndarray, lib_emb: np.ndarray, k: int, device, q_batch: int = 512, l_chunk: int = 200_000):
    """Top-``k`` library neighbours per query by cosine (unit vectors): ``(sims, idx)``, each ``(n_query, k)``."""
    k = min(k, len(lib_emb))
    sims_out, idx_out = [], []
    for i in range(0, len(query_emb), q_batch):
        q = torch.from_numpy(query_emb[i : i + q_batch].astype(np.float32)).to(device)
        best_s = best_i = None
        for j in range(0, len(lib_emb), l_chunk):
            block = torch.from_numpy(lib_emb[j : j + l_chunk].astype(np.float32)).to(device)
            s, ix = (q @ block.T).topk(min(k, block.shape[0]), dim=1)
            ix = ix + j
            if best_s is None:
                best_s, best_i = s, ix
            else:
                best_s, pos = torch.cat([best_s, s], dim=1).topk(k, dim=1)
                best_i = torch.cat([best_i, ix], dim=1).gather(1, pos)
        sims_out.append(best_s.cpu().numpy())
        idx_out.append(best_i.cpu().numpy())
    return np.concatenate(sims_out), np.concatenate(idx_out)


def consensus_fingerprints(
    spectrum_molecule: np.ndarray,
    sims: np.ndarray,
    idx: np.ndarray,
    lib_struct: np.ndarray,
    pool_fp_packed: np.ndarray,
    k: int = 50,
    beta: float = 20.0,
) -> tuple[dict, dict]:
    """One consensus fingerprint per molecule from its spectra's nearest library spectra.

    ``spectrum_molecule[i]`` is the molecule of query spectrum ``i``; ``lib_struct[j]`` the pool index of library
    spectrum ``j``. Neighbour weights are ``softmax(beta * cosine)`` over all of the molecule's neighbours, summed per
    structure so a structure with many close spectra counts once with the combined weight.
    Returns ``(consensus, top_structures)``: ``{molecule: (n_bits,) float32}`` and ``{molecule: [(pool idx, weight)]}``.
    """
    consensus, tops = {}, {}
    order = pd.Series(np.arange(len(spectrum_molecule))).groupby(np.asarray(spectrum_molecule), sort=False).indices
    for mol, rows in order.items():
        s = sims[rows, :k].ravel()
        structs = lib_struct[idx[rows, :k].ravel()]
        w = np.exp(beta * (s - s.max()))
        uniq, inv = np.unique(structs, return_inverse=True)
        weight = np.bincount(inv, weights=w, minlength=len(uniq))
        weight = weight / weight.sum()
        bits = np.unpackbits(pool_fp_packed[uniq], axis=1).astype(np.float32)
        consensus[mol] = weight @ bits
        top = np.argsort(-weight)[:5]
        tops[mol] = [(int(uniq[t]), float(weight[t])) for t in top]
    return consensus, tops


soft_tanimoto = F.soft_tanimoto


def rank_with_retrieval(
    cp, pool: pd.DataFrame, pool_emb: np.ndarray, pool_fp_packed: np.ndarray, mol_ids, mol_emb, masses,
    consensus: dict, ppm: float, lam: float, scale: float,
) -> dict:  # fmt: skip
    """Rank mass-window candidates by ``scale * cosine + lam * tanimoto(candidate, consensus)``."""
    return C.rank_contrastive(
        cp, pool, pool_emb, mol_ids, mol_emb, masses, ppm,
        scale=scale, consensus=consensus, retrieval_weight=lam, pool_fp_packed=pool_fp_packed,
    )  # fmt: skip


def library_state(model, proto: team.Protocol, data_cfg: dict, device, log=print) -> tuple[np.ndarray, np.ndarray]:
    """Library embeddings and the pool index of each library spectrum, cached on the protocol object."""
    cached = getattr(proto, "_retrieval_library", None)
    if cached is not None:
        return cached
    rows = library_rows(proto, data_cfg)
    lib_emb = embed_library(model, proto, rows, F.spec_args(data_cfg), device)
    lib_struct = rows.inchikey14.astype(str).map(proto.key_index).to_numpy()
    log(f"retrieval library: {lib_emb.shape[0]:,} spectra embedded")
    proto._retrieval_library = (lib_emb, lib_struct)
    return proto._retrieval_library


def consensus_for(model, proto, spec_emb, spectrum_molecule, data_cfg: dict, retrieval_cfg: dict, device) -> dict:
    """Consensus fingerprints for the queries whose per-spectrum embeddings are ``spec_emb``."""
    lib_emb, lib_struct = library_state(model, proto, data_cfg, device)
    k = int(retrieval_cfg.get("k", 20))
    sims, idx = knn(spec_emb, lib_emb, k, device)
    consensus, _ = consensus_fingerprints(
        spectrum_molecule, sims, idx, lib_struct, proto.pool_fp, k=k, beta=float(retrieval_cfg.get("beta", 30.0))
    )
    return consensus
