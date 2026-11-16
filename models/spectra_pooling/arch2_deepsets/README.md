# DeepSets pooling

**Owner:** Member 7  |  **Schedule:** week 3  |  **model_id:** `spectra_pooling.arch2_deepsets`

## Idea
Permutation-invariant DeepSets aggregation over a query's spectra.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/spectra_pooling.arch2_deepsets/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/spectra_pooling/arch2_deepsets/train.py --config models/spectra_pooling/arch2_deepsets/config.yaml
python models/spectra_pooling/arch2_deepsets/predict.py --config models/spectra_pooling/arch2_deepsets/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
