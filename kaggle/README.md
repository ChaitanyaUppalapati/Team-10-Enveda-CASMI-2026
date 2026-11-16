# kaggle/

**Owner:** Member 6 (except `datasets/coconut/`, owned by Member 4).

The final pipeline runs **offline** in a Kaggle notebook within 9 hours. Code and weights reach
Kaggle as Kaggle Datasets:

| Folder | What | Owner |
|--------|------|-------|
| `notebooks/casmi_offline_inference.ipynb` | Submission notebook: installs `casmi` from the code dataset, runs the selected pipeline, writes `submission.csv` | Member 6 |
| `datasets/casmi-code/` | Builds the `casmi` wheel + `configs/` + `data/splits/` + selected weights into a dataset | Member 6 |
| `datasets/coconut/` | COCONUT snapshot as a Kaggle Dataset | Member 4 |

Weekly cadence (Member 6): rebuild the code dataset, run the notebook, submit, then save a
leaderboard screenshot to `experiments/leaderboard/` and add an experiment entry.
