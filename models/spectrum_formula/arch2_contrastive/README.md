# Contrastive spectrum-molecule model

**Owner:** Member 4  |  **Schedule:** week 3  |  **model_id:** `spectrum_formula.arch2_contrastive`

## Idea
Joint embedding of spectra and candidate molecules trained with a contrastive loss.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/spectrum_formula.arch2_contrastive/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/spectrum_formula/arch2_contrastive/train.py --config models/spectrum_formula/arch2_contrastive/config.yaml
python models/spectrum_formula/arch2_contrastive/predict.py --config models/spectrum_formula/arch2_contrastive/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
