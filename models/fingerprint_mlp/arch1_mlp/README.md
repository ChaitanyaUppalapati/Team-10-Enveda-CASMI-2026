# MLP fingerprint predictor

**Owner:** Member 1  |  **Schedule:** week 2  |  **model_id:** `fingerprint_mlp.arch1_mlp`

## Idea
Binned spectrum -> MLP -> predicted Morgan fingerprint; rank candidates via casmi.retrieval.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/fingerprint_mlp.arch1_mlp/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/fingerprint_mlp/arch1_mlp/train.py --config models/fingerprint_mlp/arch1_mlp/config.yaml
python models/fingerprint_mlp/arch1_mlp/predict.py --config models/fingerprint_mlp/arch1_mlp/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
