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

## Code
- Notebook: `notebooks/03_residual_mlp.ipynb` (trains on Colab, predicts offline on Kaggle from the uploaded `artifact/` dataset; its last cell exports results in this repo's layout).
- Shared pipeline: [`../common/casmi_pipeline.py`](../common/casmi_pipeline.py).
- Run outputs: `results/residual_mlp_20261123_053259/` (metrics, per-molecule ranks, training history, figures, log, submission; model weights are not committed).

## How to run
```bash
python models/spectrum_encoders/arch1_residual_mlp/train.py --config models/spectrum_encoders/arch1_residual_mlp/config.yaml
python models/spectrum_encoders/arch1_residual_mlp/predict.py --config models/spectrum_encoders/arch1_residual_mlp/config.yaml --split val
```

## Results

Validation uses Member 3's own structure-level split (`make_split`, salt `casmi-split-v1`, 2,912 molecules), not the team's frozen split, which doesn't exist yet. Rerun on the frozen split when Member 5 publishes it. LB is the public Kaggle leaderboard.

| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
| v1 | 2026-11-23 | 0.7141 | 0.138 | width 2048, depth 4, 15 epochs, 12.7 min on L4; run `residual_mlp_20261123_053259` |

### Per-class val MRR@25
| class | n | MRR@25 | top-1 | top-25 | 95% CI |
|-------|---|--------|-------|--------|--------|
| class 1 | 1413 | 0.7148 | 0.6051 | 0.9568 | 0.6958-0.7344 |
| class 2 | 1499 | 0.7133 | 0.6044 | 0.9520 | 0.6952-0.7321 |
| np proxy | 250 | 0.6824 | 0.5480 | 0.9800 | 0.6323-0.7294 |
| balanced | 2912 | 0.7141 | 0.6048 | 0.9544 | 0.6955-0.7332 |

**LB vs. validation:** every Member 3 method scores 0.10-0.15 on the LB, whatever its validation score. Validation always keeps the true structure in the candidate pool (training structures only); on the test set the true structure is usually absent from that pool, so the LB is capped by candidate coverage (estimated ~19% of test molecules, from LB / val class 2 MRR). Larger candidate pools (COCONUT, PubChem) are the main lever.
