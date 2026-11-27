# models/spectra_pooling

**Owner:** Member 7. From pooling a molecule's spectra to a full offline Kaggle pipeline.

A test molecule has 1–16 spectra (median 3), and the submission needs **one** ranked list of 25 structures per
molecule. This folder holds three architectures built on top of each other, and the offline Kaggle notebooks that
train, validate and submit them in one run.

**Best public leaderboard: 0.343 (v10)**, 2.3× the B2 library-search baseline (0.146) and the team's best.

| Architecture | What | Kaggle versions | Best LB |
|---|---|---|---|
| 1. Pooling across spectra | Member 3's ResidualMLP per spectrum, then mean / tuned mean / gated attention / DeepSets pooling of the fingerprint predictions | v1, v2, v5 | 0.247 |
| 2. Line-up reranker | LightGBM LambdaRank over each molecule's top 50–100 candidates, on spectrum-versus-structure evidence | v7, v8 | 0.281 |
| 3. Contrastive peak transformer | peak tokens → transformer → gated attention trained jointly → CLIP-style InfoNCE against a molecule encoder, with same-mass isomers as hard negatives | v9, v10 | **0.343** |

## Results

Validation is exploratory: Member 3's structure-level split `casmi-split-v1` (2,912 held-out molecules; every
spectrum of a held-out structure is removed from the search library), balanced MRR@25 over class 1 and class 2.
"NP" is the 250 `enveda-np-examples` natural products, the closest stand-in for the test. All runs: Kaggle, 2× T4,
internet off, train + validate + predict in one notebook.

| Version | Change | Validation | NP | Public LB |
|---|---|---|---|---|
| v1 | ResidualMLP + tuned-mean pooling; 276k training structures as candidates | 0.777* | 0.757* | 0.145 |
| v2 | + COCONUT + PubChemLite candidates (1.1M), gated B2 | 0.729 | 0.636 | 0.247 |
| v5 | + every PubChem structure in each 10 ppm window (10.5M) | 0.609 | 0.663 | 0.236 ✗ |
| v7 | Architecture 2: LambdaRank line-up reranker | 0.842 | 0.694 | 0.281 |
| v8 | + PubChem documentation prior, PubChem channel | 0.879 | 0.837 | 0.276 ✗ |
| v9 | Architecture 3: contrastive peak transformer; synthetic-library (enveda-180) test penalty | 0.868 | 0.723 | 0.326 |
| **v10** | two transformer seeds, enveda-180 half-weighted in training, tuned MLP/transformer fusion | **0.874** | **0.734** | **0.343** |

\*v1's pool held only training structures, so its validation is not comparable with the rest. ✗ = failed approach.

First stages compared under the same split, pool and reranker (v10 run):

| First stage | Balanced | Class 1 | Class 2 | NP | Isomer-heavy† |
|---|---|---|---|---|---|
| ResidualMLP + mean pooling | 0.703 | 0.712 | 0.695 | 0.644 | 0.682 |
| ResidualMLP + tuned mean (Arch 1) | 0.727 | 0.746 | 0.709 | 0.659 | 0.707 |
| ResidualMLP + gated attention (Arch 1) | 0.729 | 0.751 | 0.707 | 0.657 | 0.710 |
| Transformer, mean pooling | 0.790 | 0.820 | 0.760 | 0.607 | 0.766 |
| **Transformer, attention (Arch 3)** | **0.796** | 0.827 | 0.766 | 0.625 | **0.773** |
| MLP + transformer fused | 0.825 | 0.854 | 0.796 | 0.694 | 0.809 |
| + reranker (Arch 2) | **0.874** | 0.877 | 0.871 | 0.734 | 0.862 |

†The answer has 5 or more same-formula competitors within 10 ppm (80% of validation molecules).

## The v10 pipeline, step by step

| Step | What happens | Time on 2× T4 |
|---|---|---|
| 1. Candidates | training structures + COCONUT + PubChemLite, standardised with RDKit (`common/candidates.py`): 1,097,770 structures; Morgan fingerprints (2,048 bits) | ~10 min |
| 2. Stage A | Member 3's ResidualMLP trained to predict fingerprints from binned spectra (30 epochs, BCE) | ~50 min |
| 3. Stage B | pooling heads on the MLP's embeddings (mean, tuned mean, gated attention) | ~10 min |
| 4. **Architecture 3** | two contrastive transformers, one per GPU in parallel (below) | ~40 min |
| 5. First stage | per molecule, every candidate within 10 ppm scored by MLP + transformer (z-scores, transformer weight tuned on validation: 2.0) | ~2 min |
| 6. **Architecture 2** | LambdaRank over the top 100: fingerprint likelihood, fragment-formula and neutral-loss evidence (`common/formula.py`), bond-cleavage fragments, library match, contrastive scores (`common/rerank.py`); out-of-fold on validation, 3-seed model for the test | ~30 min |
| 7. Synthetic penalty | test lists only (below) | seconds |
| 8. Submission | tautomer-safe deduplication on the official InChIKey14, `submission.csv` | ~1 min |

## Architecture 3: how the transformer is trained (`common/contrastive.py`)

Why: after reranking, 70–90% of the validation score we still lose is the answer beaten by a **same-formula
isomer**. Fingerprint prediction cannot fix that: close isomers share almost every bit, so "predict the bits, then
compare" cannot separate them. Architecture 3 learns the comparison itself.

- **Spectrum encoder.** The 64 most intense peaks of a spectrum become tokens: exact m/z and neutral loss
  (precursor − m/z) as Fourier features with wavelengths from 0.01 to 1,000 Da, plus intensity. A summary token
  carries the precursor mass and adduct. 4-layer transformer, d = 256, 8 heads, pre-norm. Nothing is binned.
- **Pooling.** Gated attention (Ilse et al., 2018) over a molecule's spectra, with the scoring layer initialised to
  zero, so it starts as exact mean pooling and only moves away if that helps. It is trained together with the
  encoder (Architecture 1 only had a frozen encoder).
- **Molecule encoder.** An MLP on the candidate's Morgan fingerprint bits. Both sides are projected to 256-d and
  L2-normalised; the score is cosine similarity times a learned temperature (CLIP-style).
- **Objective.** InfoNCE: each training molecule must pick its own structure over (a) the other molecules in the
  batch and (b) **31 hard negatives drawn from its own ±10 ppm window of the 1.1M pool**, which are mostly isomers.
  A small fingerprint head on the pooled vector (BCE, weight 0.3) keeps the embedding chemically grounded.
- **Training data.** 247,666 training structures, up to 16 spectra each; each step samples a test-like set of 1–8
  spectra per molecule (median 3), 256 molecules per batch. In v10 the synthetic enveda-180 molecules (66% of
  training structures) are drawn at 50% per epoch, so natural products weigh more: the transformer's NP score rose
  from 0.564 (v9) to 0.625.
- **Schedule.** AdamW, learning rate 3e-4 with 5% warm-up and a cosine decay over the planned steps or the time
  budget, whichever ends first; fp16 on T4; up to 18 epochs, best epoch kept by validation MRR@25 on 600 molecules.
- **Two seeds.** v10 trains two models at once, one per T4, and averages their scores (single seeds 0.776 / 0.765).
- **7.5M parameters per model**, ~40 min per seed.

It is used three ways, each measured: as its own first stage (0.796 vs the MLP's 0.727), fused with the MLP (0.825),
and as reranker features, where its similarity gap is the strongest feature by gain.

## The synthetic-library penalty (enveda-180)

The hidden test comes from biological extracts. The largest training library, `enveda-180` (182,941 of 275,810
structures), is a synthetic, drug-like compound library: it contains none of the 250 natural-product examples. Our
models learn its chemistry, and before the penalty **320 of our 400 test top-1 picks were enveda-180 structures**.

Inside each **test** top 25 (nothing is dropped, the order is otherwise kept):
- **tier A**, enveda-180 structures that are not natural products (not in COCONUT, not an NP example), go to the
  end, rank 1 included;
- **tier B**, other training structures that are not natural products, go after the unflagged ones on ranks 2–25.

It cannot be tuned on validation, where 37% of answers are enveda-180 structures; there it would only look harmful.
It is a fixed domain rule. Together with the transformer it took the leaderboard from 0.281 (v7) to 0.326 (v9);
the share of each was not measured separately (a run with `POOL_SYN_DEMOTE=0` would split it).

## Credits: public Kaggle notebooks

We studied a public Kaggle notebook for this competition (the "CASMI26 v3" stack, public LB about 0.509–0.511).
**We did not reproduce or copy it**; our pipeline, models and code are our own. Two ideas in our notebooks come
from the public CASMI 2026 notebooks and are credited in the notebook text:
- **demoting enveda-180 synthetic structures in the ranking** (their two-tier "E180X" rule), which helped our rank
  (v9 onwards);
- a **PubChem documentation prior** (`ln(1 + SIDs) + ln(1 + PMIDs)`), tried in v8, which did not help us and is off
  from v9.

What we added ourselves is the contrastive peak transformer and how it is trained (above), the line-up reranker and
its features, and the pooling study. External data: COCONUT (CC0), PubChemLite (CC BY 4.0), PubChem, RDKit (BSD);
all freely available, as the competition requires. No competition data is in this repository.

## How to run (Kaggle)

The notebooks are self-contained: each writes the `common/` modules it needs and Member 3's
`casmi_pipeline.py` into the Kaggle session with `%%writefile`, then trains, validates and writes `submission.csv`.
`common/` is the source of truth; the notebooks embed it.

1. Import `notebooks/pooling_kaggle_v10.ipynb` into a Kaggle notebook.
2. Attach 4 inputs: the competition data; an RDKit 2026.03.3 cp313 wheel (`rdkit_cp313`); COCONUT
   `coconut_csv_lite` (Nov 2026) as `coconut-2026-11-lite`; PubChemLite exposomics (Sep 2026) as
   `pubchemlite-2026-09`. (v8 also needs PubChem CID-Mass / CID-SMILES / CID-SID / CID-PMID.)
3. Accelerator **GPU T4 ×2**, Internet **off**, then Save & Run All (~2.5 h; the limit is 9 h).
4. Check the log: `contrastive transformer: 7.5M parameters x 2 seeds`, alternating `seed 0:` / `seed 1:` epoch lines,
   `first stage, validation MRR@25: {...}`, the reranker ablation table, `synthetic penalty on reranked: …`, and
   `Submission uses: reranked+syn`.
5. Submit. Outputs: `submission.csv`, `m7_results/` (validation tables, ablation, loss study, `summary.json`),
   `m7_artifact/` (weights).

Settings are environment variables read in the config cell (defaults are the submitted run), for example
`POOL_CLIP=0` (Architecture 3 off), `POOL_SYN_DEMOTE=0` (penalty off), `POOL_CLIP_SEEDS=0,1`,
`POOL_CLIP_E180_KEEP=0.5`, `POOL_RERANK_K=100`. `POOL_QUICK=1` with small `POOL_CLIP_*` values runs a smoke test.
Every optional stage fails safe: if it errors, the run continues with the previous stage's lists.

## Code

| File | What |
|---|---|
| `common/contrastive.py` | **Architecture 3**: peak transformer, gated attention pooling, molecule encoder, InfoNCE with same-mass hard negatives, two-GPU training helpers |
| `common/rerank.py` | **Architecture 2**: line-up features and LambdaRank fitting (cross-fitted, seed-bagged) |
| `common/formula.py` | fragment-formula explanation: how much of a spectrum a candidate's formula can explain |
| `common/candidates.py` | external candidates (COCONUT, PubChemLite, PubChem) parsed with RDKit in parallel; library scores; gated B2; tautomer-safe deduplication |
| `common/full.py` | Architecture 1 on Member 3's pipeline: embeddings, tuned-mean control, pooled ranking, bootstrap, artifact |
| `common/pooling.py`, `common/sets.py` | pooling heads (mean, DeepSets, attention); set sampling and padding |
| `notebooks/pooling_kaggle.ipynb`, `_v2` … `_v10` | the Kaggle versions in the results table (v3, v4, v6 are intermediate builds) |
| `notebooks/pooling_comparison_colab.ipynb`, `pooling_full.ipynb` | the first pooling prototype and full Colab run |
| `tests/test_pooling.py`, `tests/test_contrastive.py` | 27 tests: pooling, sampling, features, peak selection, hard negatives, attention = mean at init, order invariance, a learnable toy task |

Run the tests with `pytest models/spectra_pooling/tests` (needs `torch`; skipped otherwise).

## What we learned

1. **Candidate coverage caps the score.** LB ≈ coverage × validation MRR held from v1 (predicted 0.141, got 0.145).
2. **More candidates can lower it** (v5): extra structures are look-alikes unless ranking can hold them down.
3. **Library-standard validation can mislead** (v8): the documentation prior won on validation and lost on the LB.
4. **Validation gains reach the LB at about a third** (v2 → v7), except where the gain is on natural products:
   v9 → v10 gained +0.011 NP on validation and +0.017 on the LB.
5. **The objective mattered more than the architecture.** A transformer trained to predict bits (Member 3) lost to
   the MLP; trained contrastively against isomers, it beat it by +0.07.
6. **Mean pooling is hard to beat with a frozen encoder**; trained jointly, attention helps a little (0.796 vs 0.790).
7. **Domain shift is large**: two thirds of training structures are synthetic, the test is natural products.

## Next steps

A validation built from the NP examples treated as unknown structures; a fusion weight above 2 (v10 chose the edge of
its grid); more seeds and epochs; a measured ablation of the synthetic penalty; larger candidate pools only together
with stronger ranking; natural-product priors learned from structure.
