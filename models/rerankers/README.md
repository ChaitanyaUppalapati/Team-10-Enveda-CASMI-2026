# models/rerankers

**Owner:** Member 5. Candidate rerankers (consume other models' scores).

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_lambdarank`](arch1_lambdarank/) | LightGBM LambdaRank reranker on B2 features | week 2 |
| [`arch2_neural_pairwise`](arch2_neural_pairwise/) | Neural pairwise reranker | week 3 |

Shared code between the two architectures goes in `common/` inside this folder (create it
if needed). Don't import from other members' model folders; read their score files instead.
