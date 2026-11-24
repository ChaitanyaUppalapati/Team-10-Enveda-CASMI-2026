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

## Code
- Notebook: `notebooks/04_peak_set_transformer.ipynb` (trains on Colab, predicts offline on Kaggle from the uploaded `artifact/` dataset; its last cell exports results in this repo's layout).
- Shared pipeline: [`../common/casmi_pipeline.py`](../common/casmi_pipeline.py).
- Run outputs: `results/peak_set_transformer_20261124_043250/` (metrics, per-molecule ranks, training history, figures, log, submission; model weights are not committed).

## How to run
```bash
python models/spectrum_encoders/arch2_peakset_transformer/train.py --config models/spectrum_encoders/arch2_peakset_transformer/config.yaml
python models/spectrum_encoders/arch2_peakset_transformer/predict.py --config models/spectrum_encoders/arch2_peakset_transformer/config.yaml --split val
```

## Results

Validation uses Member 3's own structure-level split (`make_split`, salt `casmi-split-v1`, 2,912 molecules), not the team's frozen split, which doesn't exist yet. Rerun on the frozen split when Member 5 publishes it. LB is the public Kaggle leaderboard.

| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
| v1 | 2026-11-24 | 0.6805 | 0.121 | d_model 256, 4 layers, 10 epochs, 125 min on T4; run `peak_set_transformer_20261124_043250` |

### Per-class val MRR@25
| class | n | MRR@25 | top-1 | top-25 | 95% CI |
|-------|---|--------|-------|--------|--------|
| class 1 | 1413 | 0.6837 | 0.5584 | 0.9590 | 0.6641-0.7035 |
| class 2 | 1499 | 0.6773 | 0.5490 | 0.9586 | 0.6584-0.6952 |
| np proxy | 250 | 0.6355 | 0.4880 | 0.9760 | 0.5896-0.6803 |
| balanced | 2912 | 0.6805 | 0.5537 | 0.9588 | 0.6612-0.6994 |

**LB vs. validation:** every Member 3 method scores 0.10-0.15 on the LB, whatever its validation score. Validation always keeps the true structure in the candidate pool (training structures only); on the test set the true structure is usually absent from that pool, so the LB is capped by candidate coverage (estimated ~19% of test molecules, from LB / val class 2 MRR). Larger candidate pools (COCONUT, PubChem) are the main lever.
