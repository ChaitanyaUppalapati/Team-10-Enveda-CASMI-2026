# Residual MLP fingerprint predictor

**Owner:** Member 3  |  **Schedule:** week 2  |  **model_id:** `spectrum_encoders.arch1_residual_mlp`

## Idea
Deep MLP with residual blocks and layer norm -> fingerprint head.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/spectrum_encoders.arch1_residual_mlp/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## Code
- Notebook: `notebooks/03_residual_mlp.ipynb` (trains on Colab, predicts offline on Kaggle from the uploaded `artifact/` dataset; its last cell exports results in this repo's layout).
- Shared pipeline: [`../common/casmi_pipeline.py`](../common/casmi_pipeline.py).
- Run outputs: `results/residual_mlp_20261123_053259/` (metrics, per-molecule ranks, training history, figures, log, submission; model weights are not committed).

## How to run
`train.py` and `predict.py` are still placeholders (they raise `NotImplementedError`); use the notebook.

1. **Train on Colab:** upload `notebooks/03_residual_mlp.ipynb` to Google Colab, pick a GPU runtime (T4, L4 or A100), and
   check `DRIVE_DATA_DIR` / `DRIVE_RUN_ROOT` in the first code cell. Keep `QUICK_RUN = False` for reported runs.
   **Runtime > Run all.** The run folder on Drive gets `artifact/` (weights and candidate data),
   `submission.csv`, metrics, figures and `github_export.zip`.
2. **Submit on Kaggle:** upload the files in `artifact/` as a private Kaggle Dataset. Import the same notebook
   into a Kaggle notebook for the competition, attach the competition data and **only this model's** artifact
   dataset, turn Internet off (CPU is enough), **Save & Run All**, then submit the output.
3. **Record results:** unzip `github_export.zip` at the repo root, set `lb_score` and `submitted` in the new
   `experiments/entries/` file, and copy `results/RESULTS.md` into the Results section below.

## Results

Exploratory validation: Member 3's own structure-level split (`make_split`, salt `casmi-split-v1`, 2,912 molecules). It is not the team's frozen split, so `val_mrr` in the experiment entry stays empty until the frozen split is evaluated, and this split's scores are written to `outputs/exploratory/casmi-split-v1/`. LB is the public Kaggle leaderboard.

| version | date | frozen val MRR@25 | exploratory val MRR@25 | LB MRR@25 | notes |
|---------|------|-------------------|------------------------|-----------|-------|
| v1 | 2026-11-23 | pending | 0.7141 | 0.138 | width 2048, depth 4, 15 epochs, 12.7 min on L4; run `residual_mlp_20261123_053259` |

### Per-class exploratory val MRR@25 (casmi-split-v1)
| class | n | MRR@25 | top-1 | top-25 | 95% CI |
|-------|---|--------|-------|--------|--------|
| class 1 | 1413 | 0.7148 | 0.6051 | 0.9568 | 0.6958-0.7344 |
| class 2 | 1499 | 0.7133 | 0.6044 | 0.9520 | 0.6952-0.7321 |
| np proxy | 250 | 0.6824 | 0.5480 | 0.9800 | 0.6323-0.7294 |
| balanced | 2912 | 0.7141 | 0.6048 | 0.9544 | 0.6955-0.7332 |

**LB vs. validation:** every Member 3 method scores 0.10-0.15 on the LB, whatever its validation score. Validation always keeps the true structure in the candidate pool (training structures only); on the test set the true structure is usually absent from that pool, so the LB is capped by candidate coverage (estimated ~19% of test molecules, from LB / val class 2 MRR). Larger candidate pools (COCONUT, PubChem) are the main lever.

**Tie-break:** candidates with equal scores are ordered by `inchikey14`, the same rule as `casmi.interfaces.rank_candidates`. This run predates that fix (equal scores then kept pool order); re-ranking its exported scores moves 5 rows in 2 of 2,912 validation queries, changes exploratory val MRR@25 by less than 0.0001, and changes nothing on the test set (no tied test scores), so the LB score stands.
