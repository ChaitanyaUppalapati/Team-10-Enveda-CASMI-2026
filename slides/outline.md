# Deck outline (skeleton)

**Owner:** Member 8. Target: ~15 slides, about 12-15 minutes plus demo; every member presents.
Fill the speaker column once presentation order is agreed. Final slides are due 2026-11-29.

| # | Slide | Message | Figures | Speaker |
|---|-------|---------|---------|---------|
| 1 | Title | Team 10, Enveda CASMI 2026: molecule ID from MS/MS spectra | - | |
| 2 | Problem | Rank 25 candidate structures per molecule; scored by MRR@25 on InChIKey14; offline, 9 h | - | |
| 3 | Data | Spectra, adducts, collision energies, library sources; what the audit changed | F2 | |
| 4 | Approach | Candidates -> spectral + fingerprint scoring -> pooling -> rerank -> ensemble | F1 | |
| 5 | Candidates | Missing candidates cap MRR; COCONUT coverage; PubChem ablation | F3 | |
| 6 | Validation | Frozen structure-level split; per-class reporting; why proxies mislead | - | |
| 7 | Baselines | B0 to B3, B2 is the reference for improvement | F4 | |
| 8 | Models | MLP, CNN, residual MLP, dual-branch, transformer, contrastive | F5 | |
| 9 | Ensembles | Late fusion and stacked meta-learner: what they add over the best single model | F6, F7 | M8 |
| 10 | Results | Score over time; improvement vs B2 with intervals; leaderboard evidence | F8, F9, F14 | |
| 11 | Ablations | What mattered: reweighting, pooling, pool expansion | F10, F11 | |
| 12 | Errors | Where we fail and why | F12 | |
| 13 | Reproducibility | Offline runtime and memory, repo layout, one-command reruns | F13 | |
| 14 | Limitations and failed approaches | At least one failed idea and what it taught us | - | |
| 15 | Team and lessons | Contribution table; future work | F15 | |

## Demo plan (decide by 2026-11-26)
Run the offline notebook on a few test molecules and show the top-5 candidates with scores from each model and the ensemble. Keep a recorded fallback in case the live run is slow.

## Rehearsal
Record timing per slide in `rehearsal.md` after the first run-through (2026-11-29).
