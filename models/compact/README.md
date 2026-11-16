# models/compact

**Owner:** Member 6. Compact / distilled / quantized models for the offline runtime budget.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_distilled`](arch1_distilled/) | Distilled compact predictor | week 2 |
| [`arch2_quantized`](arch2_quantized/) | Quantized compact predictor (runtime check) | week 3 |

Shared code between the two architectures goes in `common/` inside this folder (create it
if needed). Don't import from other members' model folders; read their score files instead.
