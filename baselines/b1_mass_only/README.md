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

## How to run
```bash
python baselines/b1_mass_only/train.py --config baselines/b1_mass_only/config.yaml
python baselines/b1_mass_only/predict.py --config baselines/b1_mass_only/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
