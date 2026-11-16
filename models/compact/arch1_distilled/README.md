# Distilled compact predictor

**Owner:** Member 6  |  **Schedule:** week 2  |  **model_id:** `compact.arch1_distilled`

## Idea
Small student network trained on the best teacher's predicted fingerprints.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)
- Other models' `outputs/scores/<model_id>/<split>.parquet` (list them in `config.yaml` under `inputs`)

## Output (contract)
`outputs/scores/compact.arch1_distilled/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/compact/arch1_distilled/train.py --config models/compact/arch1_distilled/config.yaml
python models/compact/arch1_distilled/predict.py --config models/compact/arch1_distilled/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
