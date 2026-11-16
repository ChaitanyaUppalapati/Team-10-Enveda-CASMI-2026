# Stacked meta-learner over all models

**Owner:** Member 8  |  **Schedule:** week 3  |  **model_id:** `ensembles.arch2_stacked_meta`

## Idea
Meta-model trained on validation scores of every model (out-of-fold).

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)
- Other models' `outputs/scores/<model_id>/<split>.parquet` (list them in `config.yaml` under `inputs`)

## Output (contract)
`outputs/scores/ensembles.arch2_stacked_meta/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/ensembles/arch2_stacked_meta/train.py --config models/ensembles/arch2_stacked_meta/config.yaml
python models/ensembles/arch2_stacked_meta/predict.py --config models/ensembles/arch2_stacked_meta/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
