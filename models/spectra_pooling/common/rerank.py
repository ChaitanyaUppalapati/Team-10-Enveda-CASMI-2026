"""Second-stage reranker: learn to pick the answer out of its own line-up of same-mass candidates.

The first stage (fingerprint model + soft Tanimoto) puts the answer in the top 25 for ~82% of validation
molecules but at rank 1 for only ~49%. This module re-orders each molecule's top-K with a learning-to-rank
model trained on line-ups: one group per validation molecule, label 1 for the answer, 0 for its same-mass
competitors. Features are spectrum-versus-structure evidence only:

  fp   fingerprint likelihood per method (not just similarity), confident-bit conflicts, list-relative scores
  ms   exact-mass evidence: fragment-formula explanation, candidate mass error, neutral losses seen in the
       spectrum (water, CO2, sugars, methoxy, ...) matched against the candidate's functional groups
  bond fragments from breaking 1-2 bonds of the candidate's actual structure, matched to the peaks
       (isomers share a formula but break differently)
  lib  how well the candidate's reference spectra in the library match (0 when it has none)
  base the first-stage score and rank

No provenance features (training library, PubChem-only, enveda-180): validation answers are always
training-library structures, so such features would teach "the answer is never PubChem-only".
The one database signal is the PubChem documentation prior (DOC_FEATURES, v8): it is its own ablation
step, and the notebook uses it only if it also lifts the natural-product examples, because library
standards are better documented than the average hidden answer (v8 scored 0.276 vs v7's 0.281 with it, so v9
leaves it off). CLIP_FEATURES (v9) are the contrastive transformer's scores: spectrum-versus-structure evidence
like the rest, added as their own ablation step.
"""

from __future__ import annotations

import os
import zlib
from functools import lru_cache
from multiprocessing import get_context

import numpy as np
import pandas as pd

from common.formula import H, explained_fraction, ion_shifts

# neutral losses: (mass, SMARTS a candidate needs to show this loss)
LOSSES = {
    "h2o": (18.010565, "[OX2H1]"),
    "co2": (43.989829, "[CX3](=O)[OX2H1,OX1-]"),
    "hcooh": (46.005479, "[CX3](=O)[OX2H1,OX1-]"),
    "nh3": (17.026549, "[NX3;H2,H1;!$(NC=O)]"),
    "ch3oh": (32.026215, "[CH3][OX2]"),
    "c2h2o": (42.010565, "[CH3]C(=O)[O,N]"),
    # pyranose rings: anomeric C1(-O), ring O, C5 carrying CH2OH / CH3 / ring CH2 / COOH, C2-C4 each with an O
    "hexose": (162.052824, "[CX4;R]1([OX2])[OX2;R][CX4;R]([CH2][OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]"),
    "dhex": (146.057909, "[CX4;R]1([OX2])[OX2;R][CX4;R]([CH3])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]"),
    "pentose": (132.042259, "[CX4;R]1([OX2])[OX2;R][CH2;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]"),
    "glca": (176.032088, "[CX4;R]1([OX2])[OX2;R][CX4;R](C(=O)[OX2H1,OX1-])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]"),
    "so3": (79.956815, "S(=O)(=O)[OX2H1,OX1-,OX2]"),
    "co": (27.994915, "[CX3]=O"),
}
GROUPS = {
    "base": ["first_score", "first_rank"],
    "fp": [],  # filled below (per method)
    "ms": ["expl_2mda", "expl_5mda", "mass_err_ppm", "loss_agree", "loss_conflict"]
          + [f"loss_{k}" for k in LOSSES],
    "bond": ["bond_expl", "bond_expl_1cut", "bond_top5"],
    "lib": ["lib_score"],
}
METHODS = ("mean_tuned", "mean", "attention_head")
for _m in METHODS:
    GROUPS["fp"] += [f"{_m}_tan", f"{_m}_ll", f"{_m}_miss_on", f"{_m}_extra_on", f"{_m}_tan_gap", f"{_m}_ll_gap"]
ALL_FEATURES = [f for g in GROUPS.values() for f in g]


# ---------- spectrum side ----------


def spectrum_evidence(rows, mz, intensity, precursors, adducts, modes, top=30, tol=0.005):
    """Peaks (as neutral-fragment targets) and the neutral losses a molecule's spectra show."""
    targets, weights, losses = [], [], np.zeros(len(LOSSES))
    loss_masses = np.array([v[0] for v in LOSSES.values()])
    for r in rows:
        inten = intensity[r]
        n = int((inten > 0).sum())
        if n == 0:
            continue
        x, w = mz[r, :n], inten[:n]
        keep = np.argsort(-w)[:top]
        x, w = x[keep], w[keep] / w[keep].sum()
        shifts = np.array(ion_shifts(adducts[r], modes[r]))
        targets.append(np.pad(x[:, None] - shifts[None, :], ((0, 0), (0, 4 - len(shifts))), constant_values=np.nan))
        weights.append(w)
        diffs = np.abs((precursors[r] - x)[:, None] - loss_masses[None, :]) <= tol  # precursor -> fragment losses
        pair = np.abs(np.abs(x[:, None] - x[None, :])[..., None] - loss_masses) <= tol  # fragment -> fragment
        losses = np.maximum(losses, (diffs.any(0) | pair.any((0, 1))).astype(float))
    if not targets:
        return None
    w = np.concatenate(weights)
    return {"targets": np.concatenate(targets), "weights": w / w.sum(), "losses": losses}


# ---------- candidate side ----------


@lru_cache(maxsize=1)
def _loss_patterns():
    from rdkit import Chem

    return [Chem.MolFromSmarts(v[1]) for v in LOSSES.values()]


@lru_cache(maxsize=400_000)
def candidate_structure(smiles, max_pairs=200):
    """(sorted fragment masses with cut counts, functional-group vector) for one SMILES, or None.

    Fragments: the pieces left after removing one acyclic bond, two bonds of the same ring, or two acyclic
    bonds (capped). Each atom keeps its implicit hydrogens; matching allows -2..+2 H shifts.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    table = Chem.GetPeriodicTable()
    atom_mass = np.array([table.GetMostCommonIsotopeMass(a.GetSymbol()) + a.GetTotalNumHs() * H
                          for a in mol.GetAtoms()])
    adj = [[] for _ in range(n)]
    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append((j, b.GetIdx()))
        adj[j].append((i, b.GetIdx()))
        bonds.append((b.GetIdx(), b.IsInRing()))

    def pieces(cut):
        seen, out = np.zeros(n, dtype=bool), []
        for s in range(n):
            if seen[s]:
                continue
            stack, total = [s], 0.0
            seen[s] = True
            while stack:
                a = stack.pop()
                total += atom_mass[a]
                for nb, bi in adj[a]:
                    if bi not in cut and not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
            out.append(total)
        return out if len(out) > 1 else []

    masses, cuts = [float(atom_mass.sum())], [0]
    acyclic = [bi for bi, ring in bonds if not ring]
    for bi in acyclic:
        for m in pieces({bi}):
            masses.append(m)
            cuts.append(1)
    for ring in mol.GetRingInfo().BondRings():
        ring = list(ring)
        for a in range(len(ring)):
            for b in range(a + 1, len(ring)):
                for m in pieces({ring[a], ring[b]}):
                    masses.append(m)
                    cuts.append(2)
    pairs = [(a, b) for k, a in enumerate(acyclic) for b in acyclic[k + 1:]][:max_pairs]
    for a, b in pairs:
        for m in pieces({a, b}):
            masses.append(m)
            cuts.append(2)
    # keep each distinct piece mass once (fewest cuts): pieces repeat a lot, and this bounds memory
    masses, cuts = np.round(np.asarray(masses), 4), np.asarray(cuts, dtype=np.int8)
    order = np.lexsort((cuts, masses))
    masses, cuts = masses[order], cuts[order]
    first = np.r_[True, masses[1:] != masses[:-1]]
    groups = np.array([float(mol.HasSubstructMatch(p)) if p is not None else 0.0 for p in _loss_patterns()])
    return masses[first], cuts[first], groups


def bond_explanation(structure, evidence, tol=0.005):
    """Share of peak weight explained by bond-cleavage pieces (any cuts / one cut), and of the top-5 peaks."""
    if structure is None or evidence is None:
        return np.nan, np.nan, np.nan
    masses, cuts, _ = structure
    t, w = evidence["targets"], evidence["weights"]
    hit_any = np.zeros(len(t), dtype=bool)
    hit_one = np.zeros(len(t), dtype=bool)
    one = masses[cuts <= 1]
    for dh in range(-2, 3):
        for col in range(t.shape[1]):
            x = t[:, col] - dh * H
            ok = np.isfinite(x)
            for arr, hit in ((masses, hit_any), (one, hit_one)):
                if len(arr) == 0:
                    continue
                i = np.searchsorted(arr, x[ok]).clip(1, len(arr) - 1)
                d = np.minimum(np.abs(arr[i] - x[ok]), np.abs(arr[i - 1] - x[ok]))
                hit[np.flatnonzero(ok)[d <= tol]] = True
    top5 = np.argsort(-w)[:5]
    return float(w[hit_any].sum()), float(w[hit_one].sum()), float(hit_any[top5].mean())


def _structures(smiles_list):
    return [candidate_structure(s) for s in smiles_list]


def precompute_structures(smiles, workers=None):
    """Fragment tables for many SMILES in parallel (fills each worker's cache, returns a dict)."""
    uniq = list(dict.fromkeys(smiles))
    chunks = [uniq[i:i + 500] for i in range(0, len(uniq), 500)]
    with get_context("fork").Pool(workers or os.cpu_count() or 1) as p:
        parts = p.map(_structures, chunks)
    return dict(zip(uniq, [s for part in parts for s in part], strict=True))


# ---------- features for one molecule's line-up ----------


def fp_features(probs, bits):
    """probs: (F,) pooled probabilities; bits: (K, F) candidate fingerprints -> dict of (K,) arrays."""
    p = np.clip(probs.astype(np.float64), 1e-4, 1 - 1e-4)
    b = bits.astype(np.float64)
    overlap = b @ p
    tan = overlap / (p.sum() + b.sum(1) - overlap + 1e-8)
    ll = (b @ np.log(p) + (1 - b) @ np.log(1 - p)) / len(p)
    return {"tan": tan, "ll": ll, "miss_on": ((p > 0.8) & (b == 0)).sum(1).astype(float),
            "extra_on": ((p < 0.05) & (b == 1)).sum(1).astype(float), "tan_gap": tan - tan.max(),
            "ll_gap": ll - ll.max()}


def lineup_features(cand, first_score, probs_by_method, pool_fp, pool_smiles, pool_formula, pool_mass, mass,
                    evidence, structures, lib_scores):
    """Feature table (K rows) for one molecule's candidates `cand` (pool indices, first-stage order)."""
    k = len(cand)
    bits = np.unpackbits(pool_fp[cand], axis=1)
    f = {"first_score": np.asarray(first_score, dtype=float), "first_rank": np.arange(k, dtype=float)}
    for m in METHODS:
        for name, v in fp_features(probs_by_method[m], bits).items():
            f[f"{m}_{name}"] = v
    peaks = None if evidence is None else (evidence["targets"], evidence["weights"])
    cache = {}
    for name, tol in (("expl_2mda", 0.002), ("expl_5mda", 0.005)):
        vals = []
        for fo in pool_formula[cand]:
            key = (fo, tol)
            if key not in cache:
                cache[key] = explained_fraction(fo, peaks, abs_tol=tol)
            vals.append(cache[key])
        f[name] = np.array(vals, dtype=float)
    f["mass_err_ppm"] = np.abs(pool_mass[cand] - mass) / mass * 1e6
    spec_loss = evidence["losses"] if evidence is not None else np.zeros(len(LOSSES))
    groups = np.array([structures[s][2] if structures.get(s) is not None else np.full(len(LOSSES), np.nan)
                       for s in pool_smiles[cand]])
    f["loss_agree"] = np.nansum(groups * spec_loss, 1)
    f["loss_conflict"] = np.nansum((1 - groups) * spec_loss, 1)
    for j, name in enumerate(LOSSES):
        f[f"loss_{name}"] = groups[:, j] * spec_loss[j]
    be = np.array([bond_explanation(structures.get(s), evidence) for s in pool_smiles[cand]])
    f["bond_expl"], f["bond_expl_1cut"], f["bond_top5"] = be[:, 0], be[:, 1], be[:, 2]
    f["lib_score"] = np.asarray(lib_scores, dtype=float)
    return pd.DataFrame(f)[ALL_FEATURES]


# ---------- learning to rank ----------


LGB_PARAMS = {"objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
              "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "lambdarank_truncation_level": 25,
              "verbose": -1}


def fit(table, features, seed=0, rounds=400):
    """LambdaRank on line-ups. table: rows grouped by 'group' (contiguous), columns features + 'label'."""
    import lightgbm as lgb

    sizes = table.groupby("group", sort=False).size().to_numpy()
    data = lgb.Dataset(table[features].to_numpy(np.float32), table["label"].to_numpy(), group=sizes,
                       feature_name=list(features), free_raw_data=False)
    return lgb.train({**LGB_PARAMS, "seed": seed}, data, num_boost_round=rounds)


def fit_bag(table, features, seeds=(0, 1, 2), rounds=400):
    """Several seeds of `fit` (row and feature bagging differ), averaged by `predict`."""
    return [fit(table, features, seed, rounds) for seed in seeds]


def predict(model, table, features):
    x = table[features].to_numpy(np.float32)
    if isinstance(model, list):
        return np.mean([m.predict(x) for m in model], axis=0)
    return model.predict(x)


# ---------- documentation prior (kept out of ALL_FEATURES: it is its own ablation step) ----------
DOC_FEATURES = ["doc_pop", "doc_known", "doc_rank", "doc_gap", "doc_iso_gap"]


def doc_features(keys, formulas, key_pop):
    """How well documented each line-up candidate is in PubChem, absolute and relative to its competitors.

    key_pop: {inchikey14: pop} for the best-documented PubChem compounds of this molecule's mass window.
    A candidate missing from it is less documented than all of them (or not in PubChem): doc_pop is nan,
    doc_known 0. doc_iso_gap compares a candidate with the best-documented candidate of the same formula,
    which is the comparison that matters when the answer loses to an isomer.
    """
    pop = np.array([key_pop.get(k, np.nan) for k in keys], dtype=float)
    known = np.isfinite(pop)
    filled = np.where(known, pop, -1.0)
    rank = np.empty(len(pop))
    rank[np.argsort(-filled, kind="stable")] = np.arange(1, len(pop) + 1)
    rank[~known] = len(pop) + 1
    top = filled.max() if len(pop) else -1.0
    iso_top = pd.Series(filled).groupby(np.asarray(formulas, dtype=object)).transform("max").to_numpy()
    return pd.DataFrame({
        "doc_pop": pop,
        "doc_known": known.astype(np.float32),
        "doc_rank": rank,
        "doc_gap": np.where(known, pop - top, np.nan),
        "doc_iso_gap": np.where(known, pop - iso_top, np.nan),
    })


# ---------- contrastive transformer (Architecture 3, common/contrastive.py; its own ablation step) ----------
CLIP_FEATURES = ["clip_sim", "clip_rank", "clip_gap", "clip_iso_gap", "clip_z"]


def clip_features(sim, formulas):
    """Contrastive score of each line-up candidate, absolute and relative to the line-up and to its isomers."""
    sim = np.asarray(sim, dtype=float)
    rank = np.empty(len(sim))
    rank[np.argsort(-sim, kind="stable")] = np.arange(1, len(sim) + 1)
    iso_top = pd.Series(sim).groupby(np.asarray(formulas, dtype=object)).transform("max").to_numpy()
    top = sim.max() if len(sim) else 0.0
    return pd.DataFrame({
        "clip_sim": sim,
        "clip_rank": rank,
        "clip_gap": sim - top,
        "clip_iso_gap": sim - iso_top,
        "clip_z": (sim - sim.mean()) / (sim.std() + 1e-6) if len(sim) else sim,
    })


def folds(groups, n=5):
    return np.array([zlib.crc32(str(g).encode()) % n for g in groups])


def cross_fit(table, features, n_folds=5, seed=0):
    """Out-of-fold scores: each molecule is scored by a model that never saw its line-up."""
    out = np.full(len(table), np.nan)
    fold = folds(table["group"].to_numpy(), n_folds)
    for f in range(n_folds):
        train = table[fold != f]
        if train.label.sum() == 0 or (fold == f).sum() == 0:
            continue
        fitted = fit(train, features, seed)
        out[fold == f] = predict(fitted, table[fold == f], features)
    return out
