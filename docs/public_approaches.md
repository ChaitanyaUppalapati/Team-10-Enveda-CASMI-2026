# Public notebooks and discussion review

**Owner:** Member 8  |  **Status:** first pass, 2026-11-23

> **Caveat.** Everything below comes from third-party GitHub READMEs found by web search. We have **not** run their code or verified their scores, and none of them states a license or credits the public Kaggle notebook (prvsiyan / haideptry, "Analog Propagation") that our proposal cites. We could not open the Kaggle notebooks or discussion threads themselves. Treat the numbers as leads, not facts. Re-check against the Kaggle pages before quoting any score in the report.
> Local validation numbers (different splits, different sizes) are **not** comparable to public-leaderboard numbers or to ours.

## Approaches found

| Source | Approach | Key ideas |
|--------|----------|-----------|
| [ibnmarzuk/casmi26-molecule-id](https://github.com/ibnmarzuk/casmi26-molecule-id) | Library search only | ±15 ppm precursor filter; peak cleaning (intensity floor, drop peaks > precursor+2 Da, top-64, sqrt); modified cosine + spectral entropy similarity; **reciprocal rank fusion across a molecule's 1-16 spectra**; fill remaining slots with mass-nearest SMILES |
| [Isaackjoshua/casmi-2026](https://github.com/Isaackjoshua/casmi-2026) | Multi-phase pipeline (most detailed) | Library search -> COCONUT pool with **analog propagation** (`max_a sim(anchor)^4 * Tanimoto(candidate, anchor)`) -> MLP spectrum-to-Morgan fingerprint, fused with propagation (alpha 0.3) -> PubChem only as slot filler -> peak transformer + MLP ensemble -> multi-fingerprint scoring (Morgan r2/r3 + MACCS, orthogonal blocks down-weighted to ~0.3) |
| [Clyde0513/CASMI-2026](https://github.com/Clyde0513/CASMI-2026) | Data plumbing + cosine library baseline | Streamed train subsets, sharded sparse caches, mass/adduct/mode-filtered cosine, molecule-level aggregation, official-style scorer. Retriever only proposes structures present in the training library |
| [mxm-1x/Enveda-CASMI-2026](https://github.com/mxm-1x/Enveda-CASMI-2026) | CPU-first, validation-driven | Structure-grouped folds; source / structure / database holdouts; COCONUT exact-mass ranking; fragment reranker; HistGradientBoosting reranker; **fill-only blend** (keep spectral rank, fill open slots) |
| kabir0774/Enveda-Casmi-2026 | not read | README URL returned 404 |

Common pattern: precursor-mass filter -> spectral similarity to a training library -> pool across a molecule's spectra -> fill remaining slots with mass-matched candidates. Learned fingerprint models are added on top by the stronger entries.

## Reported scores

Public leaderboard MRR@25 (self-reported by Isaackjoshua; the README's phase table and narrative disagree on some phase labels, so only the endpoints are reliable):

| Stage | Public LB |
|-------|-----------|
| Library search only | 0.063 |
| + COCONUT analog propagation | 0.194 |
| + MLP fingerprint fused | 0.214-0.230 (labels conflict) |
| + PubChem slot filler | 0.232 |
| + peak transformer ensemble | 0.236-0.240 |
| + multi-fingerprint scoring (best) | **0.245** |

Local validation only (not comparable to LB): mxm-1x reports 0.926 on a natural-product source holdout where the library contains the answer, and 0.000 on a stricter split with the answer removed from the library. Their closed-world COCONUT reranker benchmarks range 0.14 (exact mass) to 0.51 (HistGradientBoosting). Isaackjoshua reports ~0.49 on a 150-structure local hard validation set while the LB stayed ~0.19-0.23.

No leaderboard scores were found for ibnmarzuk or Clyde0513 (the latter's 1.0 is a self-retrieval smoke test).

## Lessons reported by others (relevant to us)

- **Library-only ceiling is low (~0.06-0.15 LB).** Candidate generation (COCONUT) and fingerprint models are where the gains came from. This matches our proposal.
- **Tighter precursor mass window helped monotonically** down to ~1 mDa (local val 0.17 -> 0.44).
- **Local proxies misled.** Tanimoto@0.5 and timsTOF-only validation pointed the wrong way; expanding the anchor library raised local val to ~0.49 while the LB stayed flat at 0.193. A natural-product validation set tracked the real outcome best. -> Our frozen split should be structure-level and we should report per-class numbers.
- **Merging PubChem into the pool hurt** (mass-matched distractors); using it only to fill empty slots was neutral-to-slightly-positive (+0.002).
- **Ensembling similar models gave no gain.** Naively summing fingerprint log-likelihood blocks fails because MACCS dominates; blocks had to be standardized within the mass window and down-weighted.
- **Hidden-test crashes happen.** One kernel crashed on the hidden test set; the fix was coverage checks that warn instead of raise, plus per-molecule try/except with a frequency fallback. Our offline notebook (Member 6) should do the same.
- Runtime is not the bottleneck: the library baseline ran the full test in ~1-2 min; MLP training was ~11 min on an RTX A4000.

## Ideas worth trying

For the ensembles (Member 8):
1. **Rank-based fusion (RRF) as a first fusion option.** Raw scores from B2 (cosine) and fingerprint models (log-likelihood) live on different scales, so normalise per query before combining.
2. **Standardise per query/mass window and down-weight correlated blocks** before summing (the MACCS-dominance lesson). Tune weights on the frozen val split, not by hand.
3. **Fill-only blending** (keep the strongest model's top ranks, fill remaining slots from others) as a safe fallback.
4. **Expect diversity to matter:** ensembling near-identical models gave nothing, so the stacker should get genuinely different inputs (library search vs. CNN vs. MLP vs. formula branch).

For the team:
- Analog propagation (Tanimoto-weighted similarity to library anchors) is the biggest single jump in the one detailed public pipeline. Worth checking whether M3/M4 plan it.
- Candidate coverage is the hard cap on MRR; report coverage separately.
- Add a try/except + fallback to the offline notebook.

## Open items

- Read the Kaggle notebooks and discussion directly (needs a Kaggle login) and replace the third-party claims above with first-hand notes.
- Confirm the public leaderboard range so we can place our B2 score in context.
