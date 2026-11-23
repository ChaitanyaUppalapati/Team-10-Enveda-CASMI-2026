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
| `notebooks/pooling_comparison_colab.ipynb` | one Run all on a Colab GPU: train the encoder, then compare mean vs DeepSets vs attention |

Run the tests with `pytest models/spectra_pooling/tests` (needs `torch`; skipped otherwise, like
Member 2's CNN tests).

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
