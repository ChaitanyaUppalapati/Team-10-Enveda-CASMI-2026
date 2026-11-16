# B3: best fingerprint model + COCONUT + reranker

**Owner:** Member 5  |  **Schedule:** week 2  |  **model_id:** `baselines.b3_integrated`

## Idea
Integration of the best fingerprint model's scores, COCONUT candidates and the reranker.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)
- Other models' `outputs/scores/<model_id>/<split>.parquet` (list them in `config.yaml` under `inputs`)

## Output (contract)
`outputs/scores/baselines.b3_integrated/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python baselines/b3_integrated/train.py --config baselines/b3_integrated/config.yaml
python baselines/b3_integrated/predict.py --config baselines/b3_integrated/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
