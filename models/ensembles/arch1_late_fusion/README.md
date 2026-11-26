# Late-fusion ensemble (spectral + fingerprint scores)

**Owner:** Member 8  |  **Schedule:** week 2  |  **model_id:** `ensembles.arch1_late_fusion`

## Idea
Weighted combination of normalised scores from B2 and fingerprint models.

## Ways to run
0. **v3, PubChem-scale recall + priors**: `notebooks/kaggle_v3_recall_priors.ipynb` (code:
   `../ensemble_common/standalone/recall.py`). v2's pipeline plus a per-molecule PubChem channel (the 105.9M-structure
   tier, scored with our own fingerprint MLP), a documentation prior `pop` and a natural-product flag `np` as fusion
   sources, and validation copies that treat each structure as unknown (not a training structure, no library
   spectra), which is the hidden test's situation. Fusion weights are tuned, cross-fitted, on
   0.2 × class 2 + 0.8 × unknown-structure copies instead of balanced MRR. The ideas come from the public CASMI 2026
   notebooks (documentation prior, PubChem channel, uniqueness rule); the code is ours and the datasets are credited
   in the notebook. **v3.1** (`notebooks/kaggle_v3_1_ce_losses.ipynb`) is the same notebook with two richer MLP
   inputs switched on: collision energy (mean, ramp width, present; `teamsplit.collision_energy`) and a
   neutral-loss block (precursor − fragment, 500 bins; `FusionConfig.n_loss_bins`). Both are off by default, so
   v2 and v3 are unchanged.
1. **Team split, standalone (v2)**: `notebooks/kaggle_fusion_team_split.ipynb`, on Kaggle with a GPU
   and internet off. It validates on **the same split as Member 3 and Member 7**: Member 3's
   `casmi_pipeline.make_split` (salt `casmi-split-v1`), `max_query_spectra=16`, `sample_molecules(..., 3000)`,
   a 10 ppm window, and Member 3's pool and `evaluate`. It trains three sources in the notebook
   (centroid library cosine, fingerprint-MLP log-likelihood, soft Tanimoto) and fuses them
   (z-score / reciprocal rank, global or gated by library confidence). It reports 2-fold cross-fitted
   per-class MRR@25 next to B1 and writes `submission.csv`. Optional COCONUT + PubChemLite candidates
   come from the same Kaggle datasets as Member 7's v2. Code: `../ensemble_common/standalone/teamsplit.py`,
   `candidates.py`, `fpnet.PackedFingerprintMLP`. Member 3's module is passed in as `cp` and never imported
   from their folder. The code dataset zip carries a copy of `baselines/b2_library_search/casmi_pipeline.py`.
2. **Own split, first version**: `notebooks/kaggle_standalone_fusion.ipynb` (run output in
   `kaggle_standalone_fusion_run.ipynb`). It uses a crc32 split with *absent* / *covered* sets. Its numbers
   are not comparable with the team's; kept for the record.
3. **Fusing other members' score files**: fill `inputs` and `truth` in `config.yaml`, then use
   `train.py` / `predict.py` (the "How to run" section below).

### Exported score files (Kaggle notebooks, `m8_results/`; on the team Drive, not in git)
- `val_scores.parquet`, `test_scores.parquet`: **raw mass-window scores** for stacking. One row per
  candidate inside each molecule's 10 ppm window, the fused `score` plus each source's score. They do not
  reproduce the submission: molecules with an empty window are absent, there is no `fill_to_k` filler or
  B1 fallback, and equal fused scores are ordered by pool position, not by InChIKey14.
- `val_ranking.parquet`, `test_ranking.parquet`: **the final ranking** (every molecule, its 25 submitted
  candidates with filler and B1 fallback, plus `rank`). `score = 1 / rank` is strictly decreasing, so
  `casmi.interfaces.rank_candidates` returns exactly the submitted order. The validation one is cross-fitted.

## Inputs (score-file mode)
- Spectra from `casmi.data` + `casmi.preprocess` (Member 2)
- Candidates from `casmi.candidates` (Member 4), targets from `casmi.fingerprints` (Member 1)
- Other models' `outputs/scores/<model_id>/<split>.parquet` (list them in `config.yaml` under `inputs`)

## Output (contract)
`outputs/scores/ensembles.arch1_late_fusion/<split>.parquet` with columns `query_id, inchikey14, score`,
written by `casmi.interfaces.write_scores`.

## How to run
```bash
python models/ensembles/arch1_late_fusion/train.py --config models/ensembles/arch1_late_fusion/config.yaml
python models/ensembles/arch1_late_fusion/predict.py --config models/ensembles/arch1_late_fusion/config.yaml --split val
```
`train.py` tunes the weights on `val` and writes `fusion.json` with `oof_val_mrr` (each query scored with
weights tuned on the other `n_splits` folds, default 5; this is the number to report) and
`in_sample_val_mrr` (optimistic, for reference only). `predict.py --split val` writes those out-of-fold scores,
never the in-sample fit. `--split test` applies the weights fitted on all of `val`.

## Results (exploratory: Member 3's `casmi-split-v1`, not yet Member 5's frozen split)
These numbers are **exploratory**. They come from Member 3's `make_split` (salt `casmi-split-v1`, 2,912
molecules, 10 ppm), not from the frozen split in `data/splits/`, so `val_mrr` stays empty in the shared
experiment log until this model is re-scored on the frozen split with `casmi.evaluation.scorer`. The committed, internet-off Kaggle version
(`casmi-m8-late-fusion-team-split`) is in `notebooks/kaggle_fusion_team_split_run.ipynb`. It ran on a Tesla T4
in 18 min end to end, with a pool of 1,097,770 structures (275,810 training + 731,583 COCONUT +
461,082 PubChemLite, deduplicated). Its small outputs (report, per-method tables, `summary.json` with the
chosen weights, MLP training history) are in `results/`, and its figures are in `slides/figures/m8/`. The
score parquets and `fp_mlp.pt` are on the team Drive, not in git.

| version | date | exploratory val MRR@25 (balanced, `casmi-split-v1`) | LB MRR@25 | notes |
|---------|------|------------|-----------|-------|
| v1 | 2026-11-24 | 0.030 absent / 0.334 covered | – | own crc32 split, not comparable |
| v2 | 2026-11-25 | **0.687** (95% CI 0.669–0.706) | **0.200** | team split, gated z-score fusion, cross-fitted |

**Reading the LB score.** Member 7 found LB ≈ (share of hidden answers in the pool, about 0.37 for this 1.1M
pool) × (MRR when the structure has no library spectra). With our class 2 MRR of 0.558 that predicts
0.37 × 0.558 ≈ 0.206, close to the 0.200 we got. Two consequences: the hidden test behaves like class 2, so
the balanced score that chose `gated_z` was the wrong target (fp_tani alone has class 2 0.648, about 0.24
by the same estimate). And candidate recall, not the model, caps the score.

### All methods, val MRR@25 (fusion rows are 2-fold cross-fitted)
| method | balanced | class 1 | class 2 | np proxy | top-1 | top-25 |
|--------|----------|---------|---------|----------|-------|--------|
| **gated_z** (submitted) | **0.687** | 0.817 | 0.558 | 0.579 | 0.585 | 0.936 |
| global_z | 0.676 | 0.795 | 0.557 | 0.553 | 0.561 | 0.935 |
| gated_rrf | 0.646 | 0.722 | 0.571 | 0.520 | 0.566 | 0.865 |
| global_rrf | 0.613 | 0.578 | 0.648 | 0.462 | 0.507 | 0.896 |
| fp_tani (single) | 0.609 | 0.571 | 0.648 | 0.459 | 0.502 | 0.893 |
| fp_ll (single) | 0.536 | 0.490 | 0.581 | 0.441 | 0.408 | 0.891 |
| library (single) | 0.443 | 0.859 | 0.026 | 0.473 | 0.403 | 0.568 |
| B1 mass only (reference) | 0.265 | 0.444 | 0.087 | 0.359 | 0.169 | 0.622 |

### Per-class val MRR@25 (gated_z)
| class | n | MRR@25 |
|-------|---|--------|
| 1 (structure in library) | 1,413 | 0.817 |
| 2 (structure removed) | 1,499 | 0.558 |
| np-examples proxy | 250 | 0.579 |

Global z weights: library 0.16, fp_ll 0.39, fp_tani 0.45. Gated weights by library-confidence bin
(conf = 0, then tertiles at 0.45 and 0.72) shift from library 0.48 in the no-evidence bin toward fp_tani
0.58–0.63 in the others. Coverage: the truth is inside the 10 ppm window for 99.1% of molecules.
Median window size is 96 candidates. Test: 400 molecules, none on the B1 fallback.
