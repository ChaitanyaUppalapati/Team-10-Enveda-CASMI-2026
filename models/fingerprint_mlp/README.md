# models/fingerprint_mlp

**Owner:** Member 1. MLP fingerprint predictors.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_mlp`](arch1_mlp/) | MLP fingerprint predictor | week 2 |
| [`arch2_mlp_adduct_ce`](arch2_mlp_adduct_ce/) | MLP with adduct and collision-energy inputs | week 3 |

Shared code between the two architectures goes in `common/` inside this folder (create it
if needed). Don't import from other members' model folders; read their score files instead.
