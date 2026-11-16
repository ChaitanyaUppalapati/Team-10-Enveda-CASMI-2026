# B0: placeholder submission

**Owner:** Member 6  |  **Schedule:** week 1  |  **model_id:** `baselines.b0_placeholder`

## Idea
Format-valid dummy ranking used to test the offline notebook and submission checker.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/baselines.b0_placeholder/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python baselines/b0_placeholder/train.py --config baselines/b0_placeholder/config.yaml
python baselines/b0_placeholder/predict.py --config baselines/b0_placeholder/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
