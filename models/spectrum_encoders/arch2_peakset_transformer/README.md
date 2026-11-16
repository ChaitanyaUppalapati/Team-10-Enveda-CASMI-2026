# Peak-set transformer encoder

**Owner:** Member 3  |  **Schedule:** week 3  |  **model_id:** `spectrum_encoders.arch2_peakset_transformer`

## Idea
Each peak (m/z, intensity) is a token; transformer encoder -> pooled -> fingerprint head.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/spectrum_encoders.arch2_peakset_transformer/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/spectrum_encoders/arch2_peakset_transformer/train.py --config models/spectrum_encoders/arch2_peakset_transformer/config.yaml
python models/spectrum_encoders/arch2_peakset_transformer/predict.py --config models/spectrum_encoders/arch2_peakset_transformer/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
