# Multi-resolution CNN

**Owner:** Member 2  |  **Schedule:** week 3  |  **model_id:** `spectrum_cnn.arch2_multires_cnn`

## Idea
Parallel CNN branches over several bin widths, fused before the fingerprint head.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/spectrum_cnn.arch2_multires_cnn/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/spectrum_cnn/arch2_multires_cnn/train.py --config models/spectrum_cnn/arch2_multires_cnn/config.yaml
python models/spectrum_cnn/arch2_multires_cnn/predict.py --config models/spectrum_cnn/arch2_multires_cnn/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
