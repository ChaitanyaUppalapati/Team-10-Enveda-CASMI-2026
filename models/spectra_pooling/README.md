# models/spectra_pooling

**Owner:** Member 7. Pooling across the multiple spectra of one query.

A test molecule has 1–16 spectra (median 3), and the submission needs **one** ranked list per
molecule. Each spectrum is encoded on its own; the pooling step combines the set into one
fingerprint prediction, which is then used to rank candidate structures.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_attention_pooling`](arch1_attention_pooling/) | Gated attention pooling: learns a weight per spectrum | week 2 |
| [`arch2_deepsets`](arch2_deepsets/) | DeepSets: rho(mean phi(h_i)), every spectrum counts equally | week 3 |
| baseline (no training) | Mean of per-spectrum fingerprint probabilities (what Member 2's CNN scorer does) | — |

## Code

| File | What |
|---|---|
| `common/pooling.py` | `SpectrumEncoder` (spectrum → embedding + fingerprint), `DeepSetsPool`, `AttentionPool`, `mean_probabilities` |
| `common/sets.py` | group spectra by molecule, draw test-like set sizes (1–16, median 3), pad sets with a mask |
| `common/evaluate.py` | Tanimoto candidate scoring to the shared `query_id, inchikey14, score` format, stand-in MRR@25 |
| `tests/test_pooling.py` | shapes, padding/order invariance, attention weights sum to 1, set sampling, MRR |
| `common/full.py` | full run on Member 3's pipeline: embeddings, `mean_tuned` fair control, pooled ranking, paired bootstrap, artifact export/load |
| `notebooks/pooling_comparison_colab.ipynb` | prototype (stand-ins): train the encoder, then compare mean vs DeepSets vs attention |
| `notebooks/pooling_full.ipynb` | **full run** (L4, ~3–4 h): Member 3's split, candidate pool, ResidualMLP and evaluation; 3 seeds; fair control; test `submission.csv`; artifact for the offline notebook |
| `common/candidates.py` | external candidates (COCONUT, PubChemLite) parsed with RDKit in parallel; gated B2 fusion with cross-fitted threshold tuning |
| `notebooks/pooling_kaggle.ipynb` | **Kaggle v1** (offline, 2× T4, ~52 min): train + validate + `submission.csv` in one run; training structures as candidates. LB **0.145** |
| `notebooks/pooling_kaggle_v2.ipynb` | **Kaggle v2** (~60 min): adds COCONUT + PubChemLite candidates (pool 276k → 1.1M) and gated B2. LB **0.247** |

Run the tests with `pytest models/spectra_pooling/tests` (needs `torch`; skipped otherwise, like
Member 2's CNN tests).

## Kaggle results

Attach as Kaggle Datasets (internet off at scoring): the competition data, an RDKit 2026.03.3 cp313 wheel,
COCONUT `coconut_csv_lite` (Nov 2026, CC0) and PubChemLite exposomics (v3.3.0, CC BY 4.0).

| Version | Candidate pool | Validation (balanced MRR@25, Member 3 split) | Public LB |
|---|---|---|---|
| v1 | 276k training structures | 0.777 (`mean_tuned`) | 0.145 |
| v2 | 1.1M (+COCONUT, +PubChemLite) | 0.729 (`gated B2`, cross-fitted) | 0.247 |

v1 matched its prediction (19% coverage x class-2 MRR = 0.141), so LB ~ coverage x validation MRR. By that
estimate v2 covers about 37% of the hidden molecules: candidate coverage, not the model, still limits the score.
Run outputs and weights are kept on Drive, not in the repo (`Results/` is ignored).

## Prototype notebook: stand-ins

The notebook gives results now without waiting on other members. Each stand-in is local to the
notebook and is swapped for the team's version when it lands; the pooling code does not change.

| Piece | Stand-in | Replace with |
|---|---|---|
| Validation split | random 10% of `inchikey14`, seed 2026 | Member 5's frozen split (`data/splits/`) |
| Candidates | training structures within 10 ppm of the **true** neutral mass (answer always present, so optimistic) | Member 4's `casmi.candidates` |
| Scorer | `common/evaluate.py` | Member 5's `casmi.evaluation.scorer` |
| Per-spectrum model | small MLP encoder trained in the notebook | Member 1 / Member 2 per-spectrum fingerprint probabilities |

All three pooling methods use the **same frozen encoder and the same candidates**, so the
comparison between them is fair even though absolute MRR is not leaderboard-comparable.

Score files are written with `casmi.interfaces.write_scores` under model ids
`spectra_pooling.{mean_baseline,arch2_deepsets,arch1_attention_pooling}.prototype`, split `val`.

## Request to other members

Pooling needs **per-spectrum** fingerprint probabilities before any averaging
(`spectrum_id, query_id, fp_0..fp_1023`). Member 2's `score_candidates` currently mean-pools
internally; exposing the per-spectrum array lets any fingerprint model plug into these poolers.
