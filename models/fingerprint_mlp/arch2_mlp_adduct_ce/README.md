# MLP with adduct and collision-energy inputs

**Owner:** Member 1  |  **Schedule:** week 3  |  **model_id:** `fingerprint_mlp.arch2_mlp_adduct_ce`

## Idea
Arch 1 plus one-hot adduct and normalised collision energy as extra inputs.

## Inputs
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)

## Output (contract)
`outputs/scores/fingerprint_mlp.arch2_mlp_adduct_ce/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/fingerprint_mlp/arch2_mlp_adduct_ce/train.py --config models/fingerprint_mlp/arch2_mlp_adduct_ce/config.yaml
python models/fingerprint_mlp/arch2_mlp_adduct_ce/predict.py --config models/fingerprint_mlp/arch2_mlp_adduct_ce/config.yaml --split val
```

## Results (frozen validation split; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
|         |      |            |           |       |

### Per-class val MRR@25
| class | n | MRR@25 |
|-------|---|--------|
