# models/

One folder per member, each with two architectures (`arch1_*` in week 2, `arch2_*` in week 3).
Every architecture folder has the same template: `README.md`, `config.yaml`, `train.py`,
`predict.py`, `notebooks/`. `tests/test_repo_layout.py` enforces it.

| Folder | Owner |
|--------|-------|
| [`fingerprint_mlp`](fingerprint_mlp/) | Member 1 |
| [`spectrum_cnn`](spectrum_cnn/) | Member 2 |
| [`spectrum_encoders`](spectrum_encoders/) | Member 3 |
| [`spectrum_formula`](spectrum_formula/) | Member 4 |
| [`rerankers`](rerankers/) | Member 5 |
| [`compact`](compact/) | Member 6 |
| [`spectra_pooling`](spectra_pooling/) | Member 7 |
| [`ensembles`](ensembles/) | Member 8 |

**Contract:** `predict.py` writes `outputs/scores/<family>.<arch>/<split>.parquet` with
`casmi.interfaces.write_scores`. Rerankers, ensembles and B3 combine models only through
these files.

To add a third architecture, copy an existing `arch*` folder, rename it `arch3_<name>`, and
set `model_id: <family>.arch3_<name>` in its `config.yaml`.
