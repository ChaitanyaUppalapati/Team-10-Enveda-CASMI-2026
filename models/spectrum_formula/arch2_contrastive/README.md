# spectrum_formula / arch2_contrastive

**Owner:** Member 4  |  **Schedule:** week 3  |  **model_id:** `spectrum_formula.arch2_contrastive`

Joint embedding of spectra and candidate molecules, trained with a contrastive loss. Unlike arch1 (which predicts a
fingerprint), this model learns a shared unit-norm space and ranks candidates by cosine similarity.

```
spectrum (0.5 Da bins + neutral-loss block) + precursor/adduct/CE -> spectrum encoder -> z_s  (d = 512, unit norm)
candidate Morgan bits (2048)                                       -> molecule encoder -> z_m  (d = 512, unit norm)
score = cos(z_s, z_m)   (a molecule's spectra are averaged before comparing)
```

- **Loss:** InfoNCE with learnable temperature. Negatives are the batch's other molecules **plus 32 hard negatives per
  spectrum**: pool structures within 10 ppm of the true structure's mass, i.e. exactly the candidates the model must
  separate at test time. Duplicate structures in a batch and hard negatives equal to the positive are masked.
- **Candidates:** the team pool (training structures + optional `data.external_pools`, e.g. COCONUT), mass-window
  filtered at 10 ppm by Member 3's helpers; lists are padded to 25 like the team's ranking.
- **Optional gate:** the gated library-search fusion (`common/b2gate.py`) works on these rankings unchanged.
- **Speed:** pool fingerprints stay on the GPU, half precision on CUDA. The hard negatives make training compute-heavy
  (about 8,000 molecules per step), so use a GPU for full data.

## Inputs
- Team protocol via `common/team.py` (Member 3's split, pool and scorer); spectra features from `common/features.py`.
- Candidates from `casmi.candidates` (Member 4).

## Output (contract)
`outputs/scores/spectrum_formula.arch2_contrastive/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`; `--split test` also writes `submission.csv`.

## How to run
```bash
python models/spectrum_formula/arch2_contrastive/train.py --config models/spectrum_formula/arch2_contrastive/config.yaml
python models/spectrum_formula/arch2_contrastive/predict.py --config models/spectrum_formula/arch2_contrastive/config.yaml --split val
```
Kaggle: `../notebooks/arch2_contrastive_kaggle.ipynb`.

## Results (team protocol; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
| v1      | 2026-11-26 | **0.8046** model only; 0.8086 gated (np proxy 0.6586) | not submitted | full data (884k spectra), 12 epochs, 1024/2048 d 512, pool = train + COCONUT, fp16 on 2x T4 (35 min) |
| v2      | 2026-11-26 | 0.8126 model only; 0.8156 gated (np proxy 0.6498) | not submitted | wider 2048/4096/4096: balanced +0.008, np proxy -0.009 vs v1, 54 min on 2x T4 |
| smoke   | 2026-11-26 | 0.6759 (400 val molecules, pool = train structures) | not submitted | 1 spectrum/structure, 2 epochs, 0.5 Da + neutral loss, 1024/2048, d 512: wiring check, not a result |

### Per-class val MRR@25
| class | n | MRR@25 (model only) | top1 | top5 | top25 |
|-------|---|---------------------|------|------|-------|
| class 1 | 1413 | 0.8458 | 0.774 | 0.937 | 0.981 |
| class 2 | 1499 | 0.7633 | 0.641 | 0.911 | 0.976 |
| np proxy | 250 | 0.6586 | 0.532 | 0.836 | 0.968 |
| balanced | 2912 | 0.8046 | 0.708 | 0.924 | 0.978 |
