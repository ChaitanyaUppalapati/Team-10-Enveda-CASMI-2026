# 1D CNN

**Owner:** Member 2 | **model_id:** `spectrum_cnn.arch1_cnn1d`

Three 1D convolution blocks and a 1024-bit fingerprint head.
The layer layout is compatible with the original notebook checkpoint.

See [shared documentation](../README.md) for environment, input columns,
prototype limitations and candidate provenance.

## Train

```bash
python models/spectrum_cnn/arch1_cnn1d/train.py --smoke
python models/spectrum_cnn/arch1_cnn1d/train.py --config models/spectrum_cnn/arch1_cnn1d/config.yaml
```

Configuration defaults to existing prototype data under `outputs/member2`,
or `CASMI_MEMBER2_DIR`. Each run writes a new directory containing `best.pt`,
`history.csv` and `run.json`. `--run-dir` may name an empty output directory.
Full training is optional; imported weights remain available at
`checkpoints/cnn1d_best_val_loss.pt`.

## Score supplied candidates

```bash
python models/spectrum_cnn/arch1_cnn1d/predict.py \
  --spectra /path/to/queries.parquet \
  --candidates /path/to/query_candidates.parquet \
  --prototype --split val
```

An external candidate table and explicit real query IDs are required; the raw
prototype Parquet has no query IDs and cannot be used as a team query table
without an explicit upstream mapping. For artificial smoke fixtures, assign
synthetic IDs explicitly and keep outputs in prototype mode.

Prototype output:
`outputs/prototype/scores/spectrum_cnn.arch1_cnn1d.prototype/val.parquet`.
The file contains exactly `query_id, inchikey14, score`, validated and written
through `casmi.interfaces.write_scores`.

Use `--checkpoint /path/to/new-run/best.pt` for a newly trained model.
Formal val output is blocked pending team frozen-split integration.

## Results

No frozen validation MRR@25 or leaderboard score is claimed here. Existing
notebook results remain prototype oracle evaluation; conversion smoke losses
are not comparable experiment results.
