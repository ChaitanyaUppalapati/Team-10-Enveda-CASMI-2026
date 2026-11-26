# B2: library search (cosine / modified cosine + pooling)

**Owner:** Member 3  |  **Schedule:** week 1  |  **model_id:** `baselines.b2_library_search`

## Idea
Match query spectra to training spectra; score candidates by best/pooled similarity. This is the reference baseline for the improvement criterion (G.6).

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/baselines.b2_library_search/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## Code
- Notebook: `notebooks/02_library_search_baseline.ipynb` (runs on Colab or Kaggle; writes the shared pipeline into its runtime).
- Shared pipeline: [`models/spectrum_encoders/common/casmi_pipeline.py`](../../models/spectrum_encoders/common/casmi_pipeline.py),
  the single copy used by B1, B2 and Member 3's models.
- Run outputs: `results/baseline_library_search_20261111_070818/` (metrics, per-molecule ranks, figures, log, submission).

## How to run
`train.py` and `predict.py` are still placeholders (they raise `NotImplementedError`); use the notebook.

1. Upload `notebooks/02_library_search_baseline.ipynb` to Google Colab (no GPU needed). The competition data must be on Google Drive
   at the path in the notebook's first code cell (`DRIVE_DATA_DIR`).
2. **Runtime > Run all.** It builds the `casmi-split-v1` validation split, runs the ablation, writes
   `submission.csv`, and saves a timestamped run folder under `DRIVE_RUN_ROOT`.
3. To submit: import the same notebook into a Kaggle notebook for the competition, attach the competition data,
   turn Internet off and **Save & Run All**, then submit the output. On Kaggle it skips validation and uses the
   settings in the first code cell (`SUBMIT_*`).

## Results

Exploratory validation: Member 3's own structure-level split (`make_split`, salt `casmi-split-v1`, 2,912 molecules). It is not the team's frozen split, so `val_mrr` in the experiment entry stays empty until the frozen split is evaluated, and this split's scores are written to `outputs/exploratory/casmi-split-v1/`. LB is the public Kaggle leaderboard.

| version | date | frozen val MRR@25 | exploratory val MRR@25 | LB MRR@25 | notes |
|---------|------|-------------------|------------------------|-----------|-------|
| B2 | 2026-11-11 | pending | 0.4837 | 0.146 | modified cosine, mean pooling, 10 ppm; run `baseline_library_search_20261111_070818` |

### Per-class exploratory val MRR@25 (casmi-split-v1)
| class | n | MRR@25 | top-1 | top-25 | 95% CI |
|-------|---|--------|-------|--------|--------|
| class 1 | 1413 | 0.8628 | 0.8054 | 0.9816 | 0.8474-0.8774 |
| class 2 | 1499 | 0.1046 | 0.0454 | 0.4103 | 0.0940-0.1149 |
| np proxy | 250 | 0.5585 | 0.4760 | 0.8920 | 0.5019-0.6127 |
| balanced | 2912 | 0.4837 | 0.4254 | 0.6960 | 0.4707-0.4962 |

**LB vs. validation:** every Member 3 method scores 0.10-0.15 on the LB, whatever its validation score. Validation always keeps the true structure in the candidate pool (training structures only); on the test set the true structure is usually absent from that pool, so the LB is capped by candidate coverage (estimated ~19% of test molecules, from LB / val class 2 MRR). Larger candidate pools (COCONUT, PubChem) are the main lever.
