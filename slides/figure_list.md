# Figure list

**Owner:** Member 8. Status: `todo` until the source file exists on `main`.

| # | Figure | Source | Owner | Slide | Status |
|---|--------|--------|-------|-------|--------|
| F1 | Pipeline diagram: spectra -> preprocess -> candidate pool (COCONUT) -> models -> score files -> ensemble -> submission | draw once; mirrors CONTRIBUTING.md section 2 | M8 | 4 | todo |
| F2 | Data overview: spectra per molecule, adduct and collision-energy mix, library/source split | `notebooks/data_audit.ipynb` | M2 | 3 | todo |
| F3 | Candidate coverage: share of val answers present in COCONUT (with / without PubChem) | M4 coverage audit + PubChem ablation | M4 | 5 | todo |
| F4 | MRR@25 per baseline (B0, B1, B2, B3) on the frozen val split | `experiments/LOG.md` | M1 / M5 | 7 | todo |
| F5 | Per-model val MRR@25 bar chart, grouped by class (class 1 / 2 / np proxy) | each model README + `casmi.evaluation.per_class` | M5 | 8 | todo |
| F6 | Late-fusion weight chart: tuned weight per input model + single-model vs fused MRR | `outputs/models/ensembles.arch1_late_fusion/fusion.json` | M8 | 9 | todo |
| F7 | Recall + priors (arch2, v3): hidden-like validation vs v2 fusion, LB 0.200 -> 0.224 | `slides/figures/m8/arch2_validation_report.png` | M8 | 9 | done |
| F8 | Score over time: val MRR and leaderboard score by submission date, B2 baseline line | `experiments/score_over_time.png` (M1) | M1 | 10 | todo |
| F9 | Improvement vs B2 with bootstrap 95% intervals | `casmi.evaluation.bootstrap` (M5) | M5 | 10 | todo |
| F10 | Ablation table: library reweighting, pooling choice, PubChem pool | `docs/ablations/README.md` | M5 | 11 | todo |
| F11 | Pooling comparison: mean vs DeepSets vs attention | `models/spectra_pooling` results | M7 | 11 | todo |
| F12 | Error analysis: example molecules the ensemble gets right / wrong, by class | `docs/error_analysis/` | M7 | 12 | todo |
| F13 | Runtime and memory vs the 9-hour limit | `docs/runtime_report.md` | M6 | 13 | todo |
| F14 | Leaderboard screenshots (first and final submission) | `experiments/leaderboard/` | M6 | 10 | todo |
| F15 | Contribution table (who built what) | `docs/contribution_table.md` | M1 | 15 | todo |

Rules: one message per figure, readable at the back of the room, same colours for the same model on every slide, and every number comes from a logged experiment entry.
