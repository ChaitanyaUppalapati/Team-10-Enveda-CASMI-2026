# LightGBM LambdaRank reranker on B2 features

**Owner:** Member 5  |  **Schedule:** week 2  |  **model_id:** `rerankers.arch1_lambdarank`

## Idea
Learning-to-rank over candidate features (B2 similarity, fingerprint scores, mass error).

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)
- Other models' `outputs/scores/<model_id>/<split>.parquet` (list them in `config.yaml` under `inputs`)

## Output (contract)
`outputs/scores/rerankers.arch1_lambdarank/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/rerankers/arch1_lambdarank/train.py --config models/rerankers/arch1_lambdarank/config.yaml
python models/rerankers/arch1_lambdarank/predict.py --config models/rerankers/arch1_lambdarank/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
