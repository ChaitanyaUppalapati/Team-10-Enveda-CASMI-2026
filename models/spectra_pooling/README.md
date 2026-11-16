# models/spectra_pooling

**Owner:** Member 7. Pooling across the multiple spectra of one query.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_attention_pooling`](arch1_attention_pooling/) | Attention pooling across a molecule's spectra | week 2 |
| [`arch2_deepsets`](arch2_deepsets/) | DeepSets pooling | week 3 |

Shared code between the two architectures goes in `common/` inside this folder (create it
if needed). Don't import from other members' model folders; read their score files instead.
