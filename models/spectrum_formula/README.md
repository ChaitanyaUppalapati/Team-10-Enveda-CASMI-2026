# models/spectrum_formula

**Owner:** Member 4. Models combining spectrum and molecular formula / molecule embeddings.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_dual_branch`](arch1_dual_branch/) | Dual-branch spectrum + formula network | week 2 |
| [`arch2_contrastive`](arch2_contrastive/) | Contrastive spectrum-molecule model | week 3 |

Shared code between the two architectures goes in `common/` inside this folder (create it
if needed). Don't import from other members' model folders; read their score files instead.
