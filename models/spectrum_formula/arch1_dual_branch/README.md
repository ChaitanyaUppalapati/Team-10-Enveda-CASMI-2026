# spectrum_formula / arch1_dual_branch

**Owner:** Member 4  |  **Schedule:** week 2  |  **model_id:** `spectrum_formula.arch1_dual_branch`

Dual-branch spectrum + precursor network that predicts a Morgan fingerprint, then ranks
mass-matched candidates (COCONUT) by similarity to it.

```
binned spectrum (1200 x 1 Da, sqrt) -> spectrum branch ─┐
precursor m/z, neutral mass, adduct,                    ├─> trunk -> fingerprint head (2048 logits, BCE)
  polarity, collision energy (+ missing flag) -> meta ──┘          -> element-count head (aux, log counts)
```

- The formula of an unknown molecule is unknown at test time, so it is an **auxiliary target** (log counts of
  C, H, N, O, S, P, F, Cl, Br, I). At scoring, predicted counts can be compared with each candidate's formula
  (`predict.formula_weight`, default 0, tune on validation).
- A molecule's spectra are merged by averaging predicted bit probabilities per query.
- Candidates: `casmi.candidates` (adduct-aware mass lookup, 10 ppm) over `candidates_dir/coconut.parquet`.

```bash
python models/spectrum_formula/arch1_dual_branch/train.py   --config models/spectrum_formula/arch1_dual_branch/config.yaml
python models/spectrum_formula/arch1_dual_branch/predict.py --split val    # prints MRR@25, writes outputs/scores/...
python models/spectrum_formula/arch1_dual_branch/predict.py --split test
```

Validation uses the team protocol (Member 3's `casmi_pipeline`, loaded by `common/team.py`), so numbers are comparable with
the other members'. The model package and Kaggle notebook are in `../kaggle/` and `../notebooks/`.

## Results (team protocol: Member 3's split, pool and scorer; add an experiment entry for every row)
| version | date | val MRR@25 | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
| v1      | 2026-11-26 | 0.2499 (own hash split, NOT comparable) | not submitted | superseded: different split and pool |
| v2      | 2026-11-26 | **0.6014** balanced | not submitted | team protocol, pool = train structures; 884k spectra, 12 epochs, 1 Da bins; still improving at epoch 12 |
| v3      | 2026-11-26 | 0.5292 balanced | **0.129** | 0.5 Da bins, 4 spectra/structure, 12 ep, pool = train + COCONUT, no gate |
| v4      | 2026-11-26 | 0.6868 gated (model only 0.6536) | **0.217** | wide 1024/2048 + neutral-loss channel + gated library search |
| v5      | 2026-11-26 | 0.7470 gated (model only 0.7355) | 0.211 | xl 2048/4096, 24 epochs; val +0.06 over v4 but LB unchanged (noise) |

### Per-class val MRR@25
| class | n | MRR@25 | top1 | top5 | top25 |
|-------|---|--------|------|------|-------|
| class 1 | 1413 | 0.6011 | 0.462 | 0.776 | 0.935 |
| class 2 | 1499 | 0.6017 | 0.472 | 0.764 | 0.923 |
| np proxy | 250 | 0.5231 | 0.344 | 0.752 | 0.944 |
| balanced | 2912 | 0.6014 | 0.467 | 0.770 | 0.929 |
