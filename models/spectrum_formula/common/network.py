"""Network between test molecules: similar spectra lend each other candidate evidence. Owner: Member 4.

Natural products come in families that share a scaffold. Two query molecules whose spectra embed close together probably
have structurally similar answers, so each molecule's candidates are re-scored by their Tanimoto similarity to the
neighbours' top candidates. The idea (linking test molecules by spectral similarity and letting neighbours lend
their top candidates) is described in several public Kaggle notebooks for this competition; this is an independent
implementation using the spectrum embeddings of our own model. NumPy only.
"""

from __future__ import annotations

import numpy as np


def neighbor_graph(mol_ids, mol_emb: np.ndarray, k: int = 5, tau: float = 0.8, power: int = 4) -> dict:
    """``{molecule: [(neighbour, weight)]}``: up to ``k`` nearest other molecules with cosine >= ``tau``.

    ``mol_emb`` holds unit vectors; the weight is ``cosine ** power`` so only close neighbours count much.
    """
    ids = list(mol_ids)
    emb = np.asarray(mol_emb, dtype=np.float32)
    n = len(ids)
    graph: dict = {m: [] for m in ids}
    if n < 2:
        return graph
    k = min(k, n - 1)
    for start in range(0, n, 2048):
        block = emb[start : start + 2048] @ emb.T
        for r in range(block.shape[0]):
            i = start + r
            row = block[r]
            row[i] = -np.inf
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            graph[ids[i]] = [(ids[j], float(row[j]) ** power) for j in top if row[j] >= tau]
    return graph


def network_scores(
    rankings: dict, graph: dict, pool_fp_packed: np.ndarray, top_k: int = 25, neighbor_top: int = 5
) -> dict:
    """``{molecule: array(len(head))}``: network score of each of the molecule's top-``top_k`` candidates.

    Score of candidate ``x`` = sum over neighbours ``n`` of ``weight_n * mean_c Tanimoto(x, c)`` where ``c`` runs over
    the neighbour's best ``neighbor_top`` candidates.
    """
    scores: dict = {}
    for mol, neighbours in graph.items():
        head = list(rankings[mol][:top_k])
        s = np.zeros(len(head), dtype=np.float32)
        if neighbours and head:
            cand = np.unpackbits(pool_fp_packed[head], axis=1).astype(np.float32)
            cand_sum = cand.sum(axis=1)
            for other, weight in neighbours:
                top = list(rankings[other][:neighbor_top])
                evidence = np.unpackbits(pool_fp_packed[top], axis=1).astype(np.float32)
                inter = cand @ evidence.T
                union = cand_sum[:, None] + evidence.sum(axis=1)[None, :] - inter
                s += weight * (inter / np.maximum(union, 1e-9)).mean(axis=1)
        scores[mol] = s
    return scores


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-9)


def rerank(rankings: dict, scores: dict, lam: float = 0.35, top_k: int = 25, lock_top: int = 0) -> dict:
    """Re-sort each molecule's top-``top_k`` by ``z(-log rank) + lam * z(network score)``.

    The first ``lock_top`` positions stay fixed (e.g. ``lock_top=1`` protects rank 1); candidates beyond ``top_k`` keep
    their order. Molecules without any network signal are left unchanged.
    """
    out = {}
    for mol, lst in rankings.items():
        net = scores.get(mol)
        head = list(lst[:top_k])
        if net is None or len(head) < 2 or not np.any(net) or net.std() == 0:
            out[mol] = list(lst)
            continue
        base = -np.log(np.arange(1, len(head) + 1, dtype=np.float32))
        final = _z(base) + lam * _z(net)
        order = np.argsort(-final[lock_top:], kind="stable") + lock_top
        out[mol] = head[:lock_top] + [head[i] for i in order] + list(lst[top_k:])
    return out


def apply_network(rankings: dict, mol_ids, mol_emb, pool_fp_packed: np.ndarray, cfg: dict) -> dict:
    """Wrapper driven by a config dict (keys: weight, tau, k, power, top_k, neighbor_top, lock_top)."""
    if not cfg or not cfg.get("weight"):
        return rankings
    graph = neighbor_graph(
        mol_ids, mol_emb, k=int(cfg.get("k", 5)), tau=float(cfg.get("tau", 0.8)), power=int(cfg.get("power", 4))
    )
    top_k = int(cfg.get("top_k", 25))
    scores = network_scores(rankings, graph, pool_fp_packed, top_k=top_k, neighbor_top=int(cfg.get("neighbor_top", 5)))
    return rerank(rankings, scores, lam=float(cfg["weight"]), top_k=top_k, lock_top=int(cfg.get("lock_top", 0)))
