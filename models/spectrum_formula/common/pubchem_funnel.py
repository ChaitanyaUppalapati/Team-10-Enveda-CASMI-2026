"""PubChem funnel helpers: mass-window fetch, key + fingerprint, ensemble scoring, merge. Owner: Member 4.

Pipeline for one molecule: mass window from the PubChem index -> InChIKey14 + Morgan bits per candidate (RDKit) ->
ensemble logit ``mean_i(scale_i * cos(spectrum_emb_i, molecule_emb_i))`` -> merge with the main pool by key and re-rank
with a documentation prior. The funnel idea (mass window, fingerprint shortlist, documentation prior) follows the public
Kaggle write-ups for this competition; this is an independent implementation on our own models.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from casmi.candidates.pubchem_index import MassIndex


def fetch_windows(index: MassIndex, masses: dict, ppm: float = 10.0) -> pd.DataFrame:
    """Candidates for every molecule: ``mol, cid, mass, formula, pubmed, smiles`` (one row per candidate)."""
    parts = []
    for mol, mass in masses.items():
        w = index.window(float(mass), ppm)
        if len(w):
            w.insert(0, "mol", mol)
            parts.append(w)
    cols = ["mol", "cid", "mass", "formula", "pubmed", "smiles"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def _key_fp_chunk(args) -> tuple[list[str], np.ndarray]:
    smiles, radius, n_bits = args
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    chooser = rdMolStandardize.LargestFragmentChooser()
    keys: list[str] = []
    bits = np.zeros((len(smiles), n_bits // 8), dtype=np.uint8)
    for i, s in enumerate(smiles):
        key = ""
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                if len(Chem.GetMolFrags(mol)) > 1:
                    mol = chooser.choose(mol)
                key = Chem.MolToInchiKey(mol)[:14]
                bits[i] = np.packbits(gen.GetFingerprintAsNumPy(mol).astype(bool))
        except Exception:  # noqa: BLE001 - one bad record must not stop a multi-million row run
            key = ""
        keys.append(key if len(key) == 14 else "")
    return keys, bits


def key_and_fingerprint(smiles, radius: int = 2, n_bits: int = 2048, workers: int = 0, chunk: int = 20_000):
    """``(keys, packed_bits)`` for SMILES: InChIKey14 (largest fragment, ``''`` if unusable) and packed Morgan bits."""
    smiles = list(smiles)
    jobs = [(smiles[i : i + chunk], radius, n_bits) for i in range(0, len(smiles), chunk)]
    if not jobs:
        return np.array([], dtype="U14"), np.zeros((0, n_bits // 8), dtype=np.uint8)
    if len(jobs) > 1 and workers != 1:
        with ProcessPoolExecutor(max_workers=workers or None) as ex:
            parts = list(ex.map(_key_fp_chunk, jobs))
    else:
        parts = [_key_fp_chunk(j) for j in jobs]
    keys = np.array([k for p in parts for k in p[0]], dtype="U14")
    return keys, np.concatenate([p[1] for p in parts])


def ensemble_logits(models, mol_embs, scales, packed_bits: np.ndarray, device, batch: int = 8192) -> np.ndarray:
    """Average of ``scale_i * cosine_i`` between each candidate and one molecule's embedding under each model.

    ``mol_embs[i]`` is the molecule's unit spectrum embedding under ``models[i]``; ``packed_bits`` holds the candidates'
    packed Morgan bits.
    """
    import torch

    out = np.zeros(len(packed_bits), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(packed_bits), batch):
            raw = np.unpackbits(packed_bits[start : start + batch], axis=1).astype(np.float32)
            bits = torch.from_numpy(raw).to(device)
            total = 0.0
            for model, emb, scale in zip(models, mol_embs, scales, strict=True):
                e = model.encode_molecule(bits)
                total = total + scale * (e @ torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device))
            out[start : start + batch] = (total / len(models)).cpu().numpy()
    return out


def merge_by_key(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge one molecule's candidate tables by ``key``: max logit, any flag, ``pubmed`` summed.

    Each frame needs ``key``; optional columns: ``logit``, ``pubmed`` (default 0), flags ``coconut`` / ``library``.
    Stereoisomers share an InChIKey14, so their PubMed counts add up (a documented skeleton).
    """
    allc = pd.concat(frames, ignore_index=True)
    for col, default in (("pubmed", 0), ("coconut", False), ("library", False)):
        allc[col] = allc[col].fillna(default) if col in allc else default
        allc[col] = allc[col].astype(type(default))
    grouped = allc.groupby("key", sort=False).agg(
        logit=("logit", "max"), pubmed=("pubmed", "sum"), coconut=("coconut", "max"), library=("library", "max")
    )
    return grouped.reset_index()


def union_rankings(
    proto, queries, arrays, models, device, index: MassIndex, coconut_keys, ppm: float = 10.0,
    weight: float = 1.0, coconut_bonus: float = 4.0, library_bonus: float = 0.0, exclude_keys=(),
    chunk: int = 100, workers: int = 0, log=print,
):  # fmt: skip
    """Rank the team pool plus PubChem's mass window per molecule; ``(rankings, extended_pool)``.

    Score = ensemble logit + ``weight * ln(1 + PubMed articles)`` + ``coconut_bonus`` (in COCONUT) + ``library_bonus``
    (structure has reference spectra). Rankings hold indices into ``extended_pool``: the team pool followed by the
    PubChem structures that reached a top 25. ``exclude_keys`` are pool structures to treat as unknown (removal tests).
    """
    import torch  # noqa: F401  (embedding helpers need it; kept out of module import for CPU-only tooling)

    from . import contrastive as C

    cp, pool = proto.cp, proto.pool
    pkeys, pmass = pool.key.to_numpy(), pool.mass.to_numpy()
    in_coco = np.array([k in coconut_keys for k in pkeys])
    has_lib = pool.n_ref.to_numpy() > 0
    skip = np.array([k in set(exclude_keys) for k in pkeys]) if len(exclude_keys) else np.zeros(len(pkeys), bool)
    specs = [C.embed_spectra(m, *arrays, device) for m in models]
    agg = [C.aggregate_embeddings(queries.molecule_id, s) for s in specs]
    mol_ids, mol_embs = agg[0][0], [a[1] for a in agg]
    pool_embs = [C.embed_pool(m, proto.pool_fp, device) for m in models]
    scales = [float(m.scale.item()) for m in models]
    masses = cp.consensus_masses(queries)
    n_pool = len(pool)
    extra_rows: dict[str, tuple] = {}  # key -> (smiles, mass)
    rankings: dict = {}
    for c0 in range(0, len(mol_ids), chunk):
        ids = mol_ids[c0 : c0 + chunk]
        win = fetch_windows(index, {m: masses[m] for m in ids}, ppm)
        keys, bits = key_and_fingerprint(win.smiles, chunk=5000, workers=workers)
        ok = keys != ""
        win = win[ok].reset_index(drop=True)
        win["key"], bits = keys[ok], bits[ok]
        groups = win.groupby("mol", sort=False).indices
        for j, mol in enumerate(ids, start=c0):
            mass = float(masses[mol])
            vecs = [e[j] for e in mol_embs]
            idx = cp.window_indices(pmass, mass, ppm)
            idx = idx[~skip[idx]]
            pl = sum(s * (e[idx].astype(np.float32) @ v) for e, s, v in zip(pool_embs, scales, vecs, strict=True))
            frames = [pd.DataFrame({"key": pkeys[idx], "logit": pl / len(models), "coconut": in_coco[idx],
                                    "library": has_lib[idx], "pool_idx": idx})]  # fmt: skip
            if mol in groups:
                g = win.iloc[groups[mol]]
                pm = g.groupby("key", sort=False).pubmed.sum()
                first = g.drop_duplicates("key")
                lg = ensemble_logits(models, vecs, scales, bits[first.index.to_numpy()], device)
                pubchem = {"key": first.key.to_numpy(), "logit": lg, "pubmed": pm.loc[first.key].to_numpy()}
                pubchem.update(smiles=first.smiles.to_numpy(), mass=first.mass.to_numpy())
                frames.append(pd.DataFrame(pubchem))
            m = merge_by_key([f.drop(columns=["pool_idx", "smiles", "mass"], errors="ignore") for f in frames])
            m["score"] = m.logit + weight * np.log1p(m.pubmed) + coconut_bonus * m.coconut + library_bonus * m.library
            top = m.nlargest(25, "score").key.to_numpy()
            where = {k: i for k, i in zip(pkeys[idx], idx, strict=True)}
            chosen = []
            for k in top:
                if k in where:
                    chosen.append(int(where[k]))
                else:  # PubChem structure outside the pool
                    if k not in extra_rows:
                        row = frames[1][frames[1].key == k].iloc[0]
                        extra_rows[k] = (row.smiles, float(row["mass"]))
                    chosen.append(("new", k))
            rankings[mol] = chosen
        log(f"union ranking {min(c0 + chunk, len(mol_ids))}/{len(mol_ids)} molecules, {len(extra_rows)} new structures")
    new_keys = list(extra_rows)
    new_index = {k: n_pool + i for i, k in enumerate(new_keys)}
    extended = pd.concat([
        pool,
        pd.DataFrame({"key": new_keys, "smiles": [extra_rows[k][0] for k in new_keys], "formula": "",
                      "mass": [extra_rows[k][1] for k in new_keys], "n_ref": 0}),
    ], ignore_index=True)  # fmt: skip
    for mol, lst in rankings.items():
        out = [new_index[x[1]] if isinstance(x, tuple) else x for x in lst]
        if len(out) < 25:  # tiny windows: pad with the nearest pool structures like the team's ranking does
            known = set(out)
            padded = cp.fill_to_k(np.array([i for i in out if i < n_pool], dtype=int), pmass, float(masses[mol]))
            out += [int(i) for i in padded if int(i) not in known][: 25 - len(out)]
        rankings[mol] = out
    return rankings, extended
