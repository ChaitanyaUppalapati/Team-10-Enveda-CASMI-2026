# models/spectrum_encoders

**Owner:** Member 3. Residual MLP and peak-set transformer encoders.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_residual_mlp`](arch1_residual_mlp/) | Residual MLP fingerprint predictor | week 2 |
| [`arch2_peakset_transformer`](arch2_peakset_transformer/) | Peak-set transformer encoder | week 3 |

Shared code between the two architectures goes in `common/` inside this folder (create it
if needed). Don't import from other members' model folders; read their score files instead.
