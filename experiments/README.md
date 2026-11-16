# experiments/

Shared experiment log, built so it never causes merge conflicts.

- `entries/`: **one YAML file per run**, named `YYYY-MM-DD_memberN_<model_id>.yaml`. Copy
  `TEMPLATE.yaml`. Add your own; never edit others'.
- `build_log.py` (Member 1): combines all entries into `LOG.md` and `score_over_time.png`.
- `leaderboard/` (Member 6): leaderboard screenshots, named `YYYY-MM-DD_<model_id>.png`.

```bash
python experiments/build_log.py
```
