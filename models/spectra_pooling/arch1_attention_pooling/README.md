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
Current prototype: `models/spectra_pooling/notebooks/pooling_comparison_colab.ipynb` (Colab GPU,
one Run all) trains this pooler next to the mean baseline and the other pooler; see
[`../README.md`](../README.md) for the stand-ins it uses. `train.py` / `predict.py` will wrap the
same `common/` code once the frozen split and candidates exist.

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
