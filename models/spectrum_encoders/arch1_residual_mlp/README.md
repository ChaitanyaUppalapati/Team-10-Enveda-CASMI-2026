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

## How to run
```bash
python models/spectrum_encoders/arch1_residual_mlp/train.py --config models/spectrum_encoders/arch1_residual_mlp/config.yaml
python models/spectrum_encoders/arch1_residual_mlp/predict.py --config models/spectrum_encoders/arch1_residual_mlp/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
