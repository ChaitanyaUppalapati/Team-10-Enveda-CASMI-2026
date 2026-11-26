"""Self-contained late-fusion pipeline: trains its own members, needs no other member's outputs.

Members: (1) cosine library search over per-structure spectrum centroids,
(2) a neural spectrum -> Morgan-fingerprint predictor. Candidates are the structures seen in
training, filtered by precursor mass. Heavy training runs on Kaggle; see arch1_late_fusion/kaggle_run.py.
"""
