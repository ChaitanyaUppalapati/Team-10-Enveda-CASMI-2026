# PubChem-scale recall + documentation priors (v3)

**Owner:** Member 8  |  **Schedule:** week 3  |  **model_id:** `ensembles.arch2_recall_priors`

## Idea
Arch1 (late fusion) scored 0.687 on the team split but **0.200** on the leaderboard. On the split the answer is
always a training structure already in the pool; on the hidden test it usually is not, so the score is capped by
**recall** (is the answer a candidate at all?). Arch2 attacks recall and ranking of unseen structures:

1. **PubChem channel.** For every molecule, take the 10 ppm window of the 105.9M-structure PubChem tier and keep
   the best 60 structures by our fingerprint MLP (log-likelihood) and 60 by a mix of MLP score and documentation
   prior. They are appended to the 1.1M pool (training structures + COCONUT + PubChemLite), giving 1.35M.
2. **Priors as fusion sources.** `pop` = ln(1 + PubChem SIDs) + ln(1 + PMIDs), how well documented a structure is;
   `np` = natural-product flag (COCONUT, NP table, PubChem tier). Fused with arch1's `library`, `fp_ll`, `fp_tani`.
3. **Hidden-like validation.** Each validation molecule is also scored as an "unknown-structure" copy: its truth
   is removed from the pool unless an external source (COCONUT, PubChemLite, the PubChem tier) has it, and it gets
   no library spectra. Weights are tuned, 2-fold cross-fitted, on
   0.2 × class 2 + 0.4 × unknown natural products + 0.4 × other unknown structures.
4. **Uniqueness rule.** Each test molecule is a different structure, so a structure that leads two molecules is
   kept only where its fused lead is stronger. Used only if it helps on validation (it did, slightly).

The ideas come from the public CASMI 2026 notebooks (documentation prior, PubChem channel, uniqueness rule); the
code is ours and the public datasets are credited in the notebook. Code: `../ensemble_common/standalone/recall.py`
plus arch1's `teamsplit.py`, `candidates.py`, `fpnet.py`.

## Ways to run
1. **Kaggle notebook (the submitted model):** `notebooks/kaggle_v3_recall_priors.ipynb`, GPU, internet off,
   about 71 min. Inputs: the competition data, the code dataset (`casmi-m8-code-v3`: this repo's
   `models/ensembles/ensemble_common` and Member 3's `casmi_pipeline.py`), COCONUT and PubChemLite, the PubChem tier
   and its prior arrays, the popularity CSV and the natural-product table. The committed run with outputs is
   `notebooks/kaggle_v3_recall_priors_run.ipynb`.
2. **v3.1:** `notebooks/kaggle_v3_1_ce_losses.ipynb`, the same notebook with two richer MLP inputs: collision
   energy (mean, ramp width, present; `teamsplit.collision_energy`) and a neutral-loss block (precursor − fragment,
   500 bins; `FusionConfig.n_loss_bins`). Not run yet.
3. **Score-file mode (stacked meta-learner):** `train.py` / `predict.py` train an out-of-fold LambdaRank or
   logistic meta-learner over other members' `outputs/scores/<model_id>/<split>.parquet`
   (`ensemble_common/stacking.py`). Fill `inputs` and `truth` in `config.yaml`. Not run yet: it needs the frozen
   split labels and members' score files.

```bash
python models/ensembles/arch2_recall_priors/train.py --config models/ensembles/arch2_recall_priors/config.yaml
python models/ensembles/arch2_recall_priors/predict.py --config models/ensembles/arch2_recall_priors/config.yaml --split val
```

## Output (contract)
Kaggle: `submission.csv`, and `m8_results/` with `test_scores.parquet` (raw window scores, every source),
`summary.json` and the validation tables. Score-file mode: `outputs/scores/ensembles.arch2_recall_priors/<split>.parquet`
with columns `query_id, inchikey14, score`, written by `casmi.interfaces.write_scores` (out-of-fold for `val`).

## Results (exploratory: Member 3's `casmi-split-v1`, not yet Member 5's frozen split)
Same 2,912 validation molecules as arch1 (class 1: 1,413, class 2: 1,499, np proxy: 250). Small outputs are in
`results/`, the figure is `slides/figures/m8/arch2_validation_report.png`; `fp_mlp.pt`, `test_scores.parquet`
and `submission.csv` are on the team Drive, not in git.

| version | date | objective (cross-fitted) | unknown-structure MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-----|-------|
| arch1 v2 (reference) | 2026-11-25 | – | – | 0.200 | late fusion, 1.1M pool |
| v3 | 2026-11-26 | **0.538** | 0.438 | **0.224** | `v3_gated_z`, 1.35M pool, uniqueness rule on |

### All methods, val MRR@25 (fusion rows 2-fold cross-fitted, all on the 1.35M pool)
| method | objective | unknown all | unknown np | class 1 | class 2 | np proxy | top-1 |
|--------|-----------|-------------|------------|---------|---------|----------|-------|
| **v3_gated_z** (submitted) | **0.538** | **0.438** | 0.734 | 0.598 | 0.401 | 0.766 | 0.387 |
| v3_global_z | 0.535 | 0.432 | 0.740 | 0.589 | 0.391 | 0.772 | 0.375 |
| v3_global_rrf | 0.436 | 0.363 | 0.601 | 0.446 | 0.300 | 0.602 | 0.248 |
| pop (single) | 0.383 | 0.212 | 0.743 | 0.339 | 0.104 | 0.743 | 0.152 |
| v2_gated_z (arch1's fusion) | 0.360 | 0.362 | 0.332 | 0.383 | 0.407 | 0.345 | 0.296 |
| fp_tani (single) | 0.356 | 0.360 | 0.324 | 0.359 | 0.407 | 0.324 | 0.284 |
| fp_ll (single) | 0.289 | 0.282 | 0.283 | 0.288 | 0.316 | 0.283 | 0.222 |
| np (single) | 0.099 | 0.075 | 0.160 | 0.120 | 0.044 | 0.160 | 0.028 |
| library (single) | 0.044 | 0.040 | 0.059 | 0.859 | 0.026 | 0.473 | 0.403 |

Unknown-structure copies of `v3_gated_z` by class: class 1 0.502, class 2 0.378, np proxy 0.734.
Weights: fp_tani 0.54, pop 0.32, np 0.10, fp_ll 0.03, library 0.02. The PubChem shortlist holds the truth for
71.8% of validation molecules; the truth is a candidate for 80.4% of unknown-structure copies (99.2% standard).
Test: 400 molecules, median 218 candidates, none on the B1 fallback.

**Reading the LB score.** The LB rose from 0.200 to 0.224 (+12%), far less than the unknown-structure score
(0.438) suggested. The gap comes from the np proxy: `pop` alone scores 0.743 on unknown np copies, because the
enveda np-examples structures are well documented in PubChem, and the objective gives those 250 molecules 40% of
the weight, which pushed `pop` to 0.32. The hidden test's answers are evidently far less documented. The honest
predictor is the unknown-structure **class 2** score (0.378), times the lower recall of the hidden set. Next step:
tune on unknown class 2 only and cap the `pop` weight.
