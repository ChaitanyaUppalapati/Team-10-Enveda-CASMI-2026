# Spectrum CNN modules

**Owner:** Member 2.

This folder contains the two notebook-compatible prototype CNNs and their shared
implementation in `common/`. No shared contracts or other members' TODO modules
are changed.

## Environment and paths

Use Python >=3.10 with the repository dependencies, PyTorch, and exactly
`rdkit==2026.3.3` (the existing `chem` extra). Development checks use pytest and
ruff. Run commands from the repository root; direct entry-point scripts also
resolve imports when invoked from another working directory.

`CASMI_MEMBER2_DIR` overrides `member2_dir` in the architecture YAML. By default,
existing inputs and imported checkpoints are read from `outputs/member2`.
The experiment files under that directory must stay untracked. Training outputs
go to a new `outputs/member2/module_runs/<model-id>.<timestamp>/`, not to the
imported checkpoint path.

## Notebook parity

Both networks use m/z <2200, bin width 1.0, square-root intensities, maximum
aggregation for peaks in the same bin, Morgan radius 2, 1024 fingerprint bits,
and `includeChirality=False`. Fingerprints are generated from the supplied
SMILES without adding a new canonicalization step, matching the notebooks.
The final scoring-safe structure split is checked before training. This split
is a prototype split, distinct from Member 5's frozen validation split.

Multi-resolution means **parallel kernel sizes 3, 7, 15 on the same binned
spectrum**, not different bin widths. Layer names match the imported checkpoints.

Training retains AdamW (lr=0.0003, weight decay=0.0001), prevalence-derived
positive weights clipped to [1,20] (unseen bits get 1), batch size 32,
8 epochs, patience 3 and min loss improvement 0.0001. The best checkpoint and
per-epoch loss/F1/Tanimoto history are written with the run config and data paths.
`--smoke` uses one epoch and two batches per split; those numbers are only a
pipeline check.

## Candidate input contract

Prediction requires two explicit Parquet inputs:

- Spectra: `query_id` (nonempty strings), `mzs`, `intensities`. The notebook
  aliases `ms2_mzs`, `ms2_normalized_intensities` are also accepted.
- Candidate pairs: `query_id`, `inchikey14` (14 uppercase letters), `smiles`
  (or `normalized_smiles`). One row per allowed query/candidate pair, with no
  duplicate pairs, nulls or missing queries.

The candidate provider owns identity consistency (SMILES and scoring key) and
mass/adduct filtering. This module scores exactly the supplied pairs: it does
not create an oracle pool, infer query IDs from true labels, filter by the
validation truth, or perform mass lookup. Unknown or uncovered query IDs are
rejected. Multiple spectra sharing a real `query_id` are mean-pooled in
probability space. Soft Tanimoto scores use candidate chunks of at most 4096;
all supplied candidates are retained, with finite scores and unique pairs.

The model calls `casmi.interfaces.write_scores` to output exactly
`query_id, inchikey14, score`. Schema compatibility alone does not establish
competition quality or frozen-split comparability.

Member 1's fingerprint/ranking utilities and Member 4's candidate lookup are
currently TODOs. The notebook-compatible chemistry and similarity adapter is
therefore local to this owned folder. It can be replaced with the shared
implementation after its owners supply compatible code. Existing checkpoint
fingerprints have 1024 bits, whereas the shared fingerprint stub defaults to
2048; do not silently change checkpoint target dimensions.

## Prediction modes

Prototype runs must pass `--prototype`. Their output is isolated at:

`outputs/prototype/scores/<model_id>.prototype/<split>.parquet`

`--output-root` changes the base but still adds the prototype subdirectory and
model-ID suffix. Formal `--split val` without `--prototype` deliberately raises
an error until frozen query IDs, candidates and the team split are connected.
With genuine upstream query IDs and supplied candidates, train/test output
uses the standard `outputs/scores/<model_id>/<split>.parquet` convention. No
official validation score is reported by this prototype conversion.

## Commands

See each architecture README for training and prediction commands. To use a
newly trained checkpoint instead of the imported notebook checkpoint, pass
`--checkpoint <run-directory>/best.pt`.

Tests are kept within this Member 2 folder:

```bash
python -m pytest models/spectrum_cnn/tests tests/test_interfaces.py
ruff check models/spectrum_cnn/common models/spectrum_cnn/arch*/train.py models/spectrum_cnn/arch*/predict.py models/spectrum_cnn/tests
```

The imported experimental data, checkpoints and smoke outputs must never be
staged. Full training, frozen validation and Kaggle runtime verification remain
separate from the small conversion checks.
