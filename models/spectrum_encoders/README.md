# models/spectrum_encoders

**Owner:** Member 3. Residual MLP and peak-set transformer encoders.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_residual_mlp`](arch1_residual_mlp/) | Residual MLP fingerprint predictor | week 2 |
| [`arch2_peakset_transformer`](arch2_peakset_transformer/) | Peak-set transformer encoder | week 3 |

## Layout

| Path | What it holds |
|------|---------------|
| [`common/casmi_pipeline.py`](common/casmi_pipeline.py) | The one shared pipeline for all Member 3 work: data loading, validation split (`make_split`, salt `casmi-split-v1`), candidate pools, B1 mass ranking, B2 spectral library search, the two fingerprint models, MRR@25 evaluation, submission writing, run logging and figures |
| `arch1_residual_mlp/notebooks/` | Residual MLP Colab/Kaggle notebook |
| `arch2_peakset_transformer/notebooks/` | Peak-set transformer Colab/Kaggle notebook |
| [`tests/`](tests/) | Unit tests for `common/casmi_pipeline.py`, plus a check that every notebook embeds it unchanged |

Baselines B1 (`baselines/b1_mass_only/`) and B2 (`baselines/b2_library_search/`) use the same
`common/casmi_pipeline.py`.

**One source of truth.** Each notebook writes the pipeline into its runtime with a
`%%writefile casmi_pipeline.py` cell, so it runs on Colab and Kaggle without the repo. That cell
must stay identical to `common/casmi_pipeline.py`; `tests/test_casmi_pipeline.py` fails if they drift.
Edit `common/casmi_pipeline.py`, then paste it into the four notebook cells.

```bash
pytest models/spectrum_encoders/tests -q
```

Don't import from other members' model folders; read their score files instead.
