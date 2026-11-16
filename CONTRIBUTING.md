# Contributing: Team 10 workflow

Owner: Member 1. The schedule spreadsheet is the source of truth for who does what. This file explains how the repo enforces it.

## 1. Ownership map

Every folder has exactly one owner. You work freely in your own folders. To change someone else's folder, open a PR and that owner reviews it (GitHub requests the review automatically through `.github/CODEOWNERS`).

| Member | GitHub handle | Owns |
|--------|-------------------------|------|
| Member 1 | `@ChaitanyaUppalapati` | repo root, CI, `src/casmi/utils/`, `src/casmi/fingerprints/`, `src/casmi/retrieval/`, `models/fingerprint_mlp/`, experiment-log tooling, `docs/contribution_table.md` |
| Member 2 | `@Ashley96yy` | `data/`, `src/casmi/data/`, `src/casmi/preprocess/`, `notebooks/` (loading + EDA), `models/spectrum_cnn/`, final `README.md` |
| Member 3 | `@member3-gh` (pending) | `baselines/b1_mass_only/`, `baselines/b2_library_search/`, `models/spectrum_encoders/` |
| Member 4 | `@ArpanaSinghSJSU` | `src/casmi/candidates/`, `kaggle/datasets/coconut/`, `models/spectrum_formula/` |
| Member 5 | `@member5-gh` (pending) | `src/casmi/evaluation/`, `data/splits/`, `baselines/b3_integrated/`, `models/rerankers/`, `docs/ablations/` |
| Member 6 | `@member6-gh` (pending) | `src/casmi/submission/`, `kaggle/` (offline notebook, code dataset), `baselines/b0_placeholder/`, `models/compact/`, `experiments/leaderboard/`, `docs/runtime_report.md` |
| Member 7 | `@rahulmohannett` | `models/spectra_pooling/`, `docs/competition_summary.md`, `docs/error_analysis/`, `report/` |
| Member 8 | `@aswin215` | `models/ensembles/`, `docs/public_approaches.md`, `slides/` |

**Pending handles:** Members 3, 5 and 6 still use placeholders. When their usernames are known, replace `@member3-gh`, `@member5-gh` and `@member6-gh` here and in `.github/CODEOWNERS`, and add them as collaborators with write access. Until then, PRs into their folders need one approval from any member.

## 2. How the pieces fit (integration contract)

```
data (M2) -> preprocess (M2) --+--> models/* and baselines/* --> outputs/scores/<model_id>/<split>.parquet
fingerprints (M1) -------------+         ^                                    |
candidates (M4) ---------------+         |                                    v
                                 rerankers (M5), pooling (M7),       evaluation (M5): MRR@25
                                 ensembles (M8) read scores files    submission (M6): submission.csv
                                                                     kaggle/ (M6): offline notebook
```

- **Integrate through files, not imports.** Every model's `predict.py` writes a *CandidateScores* table (`query_id`, `inchikey14`, `score`) with `casmi.interfaces.write_scores()`. Rerankers, ensembles, B3, the scorer and the submission writer read only that format. Never import code from another member's `models/` folder.
- **Shared library code lives in `src/casmi/`.** If you need something from another area (such as preprocessing), import it from `casmi.<area>`. If it's missing, open an issue for the owner, or open a PR into their folder for them to review.
- **`src/casmi/interfaces.py` is the contract.** Changing it can break all eight members. Changes need approval from **both** Member 1 and Member 5 and must keep `tests/test_interfaces.py` passing.

## 3. Branches and PRs

- `main` is protected: no direct pushes, CI (`lint-and-test`) must pass, and 1 approving review is required, from the code owner when the PR touches their folder.
- Branch names: `memberN/<area>-<topic>`, for example `member3/b2-modified-cosine` or `member5/scorer-tests`.
- Keep PRs small and inside your own folder. Rebase on `main` before merging: `git pull --rebase origin main`.
- Fill in the PR template checklist.

## 4. Experiment log (no merge conflicts)

Each run gets **one new file**: `experiments/entries/YYYY-MM-DD_memberN_<model_id>.yaml`, copied from `experiments/TEMPLATE.yaml`. Never edit someone else's entry. Member 1 regenerates the combined table and chart:

```bash
python experiments/build_log.py
```

This writes `experiments/LOG.md` (and `experiments/score_over_time.png` if matplotlib is installed).
Member 6 adds leaderboard screenshots to `experiments/leaderboard/` with each weekly submission.

## 5. Local setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[all]"         # or ".[dev]" for lint/tests only
pre-commit install              # strips notebook outputs, runs ruff
pytest
```

Put competition data in `data/raw/` (gitignored). To keep it elsewhere, set `CASMI_DATA_DIR`.

## 6. Rules of thumb

- Never commit data, model weights, `outputs/`, or notebook outputs. Notebooks are for exploration; logic that others depend on goes into `src/casmi/` or your model's `train.py` / `predict.py`.
- Report validation results only on the frozen split in `data/splits/` (owned by Member 5), using `casmi.evaluation.scorer`, so numbers are comparable across members.
- Seed everything with `casmi.utils.seed.set_seed()`.
- Every model must fit the 9-hour offline Kaggle limit at inference time. Member 6 checks the full pipeline.
