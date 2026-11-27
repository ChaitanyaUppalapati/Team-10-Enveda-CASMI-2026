# models/ensembles

**Owner:** Member 8. Ensembles over other models' CandidateScores.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_late_fusion`](arch1_late_fusion/) | Late-fusion ensemble (spectral + fingerprint scores) | week 2 |
| [`arch2_recall_priors`](arch2_recall_priors/) | PubChem-scale recall + documentation / natural-product priors (v3); score-file mode: stacked meta-learner | week 3 |

Shared code between the two architectures lives in `ensemble_common/` (named so it does not
clash with the other members' `common/` packages): `inputs.py` (align score files),
`normalize.py` (per-query rrf / minmax / zscore), `fusion.py` (weighted sum + weight tuning),
`stacking.py` (features, out-of-fold LambdaRank / logistic meta-learner), `metrics.py` (local MRR@25).
Don't import from other members' model folders; read their score files instead.

Both architectures need `truth` in their `config.yaml` (the frozen val labels from Member 5)
and a non-empty `inputs` list of model_ids whose val/test score files exist.
Tests: `pytest models/ensembles/tests` (synthetic data, no competition data needed).
