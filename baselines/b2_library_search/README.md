# B2: library search (cosine / modified cosine + pooling)

**Owner:** Member 3  |  **Schedule:** week 1  |  **model_id:** `baselines.b2_library_search`

## Idea
Match query spectra to training spectra; score candidates by best/pooled similarity. This is the reference baseline for the improvement criterion (G.6).

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/baselines.b2_library_search/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python baselines/b2_library_search/train.py --config baselines/b2_library_search/config.yaml
python baselines/b2_library_search/predict.py --config baselines/b2_library_search/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
