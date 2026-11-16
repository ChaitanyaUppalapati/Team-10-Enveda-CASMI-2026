# models/ensembles

**Owner:** Member 8. Ensembles over other models' CandidateScores.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_late_fusion`](arch1_late_fusion/) | Late-fusion ensemble (spectral + fingerprint scores) | week 2 |
| [`arch2_stacked_meta`](arch2_stacked_meta/) | Stacked meta-learner over all models | week 3 |

Shared code between the two architectures goes in `common/` inside this folder (create it
if needed). Don't import from other members' model folders; read their score files instead.
