# B1: mass-only baseline

**Owner:** Member 3  |  **Schedule:** week 1  |  **model_id:** `baselines.b1_mass_only`

## Idea
Rank candidates by precursor mass error only. First leaderboard submission.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/baselines.b1_mass_only/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## Code
- Notebook: `notebooks/01_mass_only_baseline.ipynb` (runs on Colab or Kaggle; writes the shared pipeline into its runtime).
- Shared pipeline: [`models/spectrum_encoders/common/casmi_pipeline.py`](../../models/spectrum_encoders/common/casmi_pipeline.py),
  the single copy used by B1, B2 and Member 3's models.
- Run outputs: `results/baseline_mass_only_20261111_070243/` (metrics, per-molecule ranks, figures, log, submission).

## How to run
```bash
python baselines/b1_mass_only/train.py --config baselines/b1_mass_only/config.yaml
python baselines/b1_mass_only/predict.py --config baselines/b1_mass_only/config.yaml --split val
```

## Results

Validation uses Member 3's own structure-level split (`make_split`, salt `casmi-split-v1`, 2,912 molecules), not the team's frozen split, which doesn't exist yet. Rerun on the frozen split when Member 5 publishes it. LB is the public Kaggle leaderboard.

| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
| B1 | 2026-11-11 | 0.2642 | 0.103 | mass only, 5 ppm, prior_then_ppm; run `baseline_mass_only_20261111_070243` |

### Per-class val MRR@25
| class | n | MRR@25 | top-1 | top-25 | 95% CI |
|-------|---|--------|-------|--------|--------|
| class 1 | 1413 | 0.4282 | 0.2916 | 0.8450 | 0.4073-0.4492 |
| class 2 | 1499 | 0.1002 | 0.0360 | 0.4510 | 0.0901-0.1097 |
| np proxy | 250 | 0.5108 | 0.4000 | 0.8840 | 0.4632-0.5634 |
| balanced | 2912 | 0.2642 | 0.1638 | 0.6480 | 0.2487-0.2794 |

**LB vs. validation:** every Member 3 method scores 0.10-0.15 on the LB, whatever its validation score. Validation always keeps the true structure in the candidate pool (training structures only); on the test set the true structure is usually absent from that pool, so the LB is capped by candidate coverage (estimated ~19% of test molecules, from LB / val class 2 MRR). Larger candidate pools (COCONUT, PubChem) are the main lever.
