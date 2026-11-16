# Attention pooling across a molecule's spectra

**Owner:** Member 7  |  **Schedule:** week 2  |  **model_id:** `spectra_pooling.arch1_attention_pooling`

## Idea
Encode each spectrum of a query, attention-pool them into one prediction.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/spectra_pooling.arch1_attention_pooling/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/spectra_pooling/arch1_attention_pooling/train.py --config models/spectra_pooling/arch1_attention_pooling/config.yaml
python models/spectra_pooling/arch1_attention_pooling/predict.py --config models/spectra_pooling/arch1_attention_pooling/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
