# Public competition data readers

**Owner:** Member 2. Shared interfaces are unchanged.

## Evidence and raw schema status

The user supplied the original competition ZIP on 2026-11-17. Its three files
were extracted without overwrite into gitignored `data/raw/`. The schema
below is now confirmed by actual Parquet metadata, and agrees with the
earlier Colab notebook audit. All 2,539,608 training rows and all 1,213 test
rows passed the public reader's ID, precursor and peak-integrity checks.

Verified training table: **2,539,608 rows, 21 row groups**.

| Recorded raw field | Recorded Arrow type | Shared representation |
| --- | --- | --- |
| ingest_lib | string | preserved |
| normalized_smiles | string | smiles, original preserved |
| inchikey / inchikey14 | string | preserved |
| molecular_formula | string | preserved |
| ionization_mode | string | preserved |
| instrument_type | string | preserved |
| adduct / adduct_orig | string | preserved |
| precursor_mz / precursor_error_ppm | double | preserved |
| ms2_mzs | list<double> | mzs |
| ms2_normalized_intensities | list<double> | intensities |
| num_peaks | int64 | preserved as source metadata |
| base_peak_intensity | double | preserved as source metadata |
| collision_energy_ev | list<double> | collision_energy, list preserved |
| collision_energy_orig / collision_energy_orig_units | string | preserved |

Verified test table: **1,213 spectra**. Columns are
`molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities,
base_peak_intensity, adduct, ionization_mode, instrument_type,
precursor_mz, collision_energy_orig, collision_energy_ev,
collision_energy_orig_units`.
Verified sample submission: **400 rows**, columns `molecule_id, smiles`.
Test spectrum_id is unique, and the 400 native molecule_id values exactly
match the submission query set. Each query has 1 to 9 spectra. Test string
fields are Arrow large_string, peak and collision-energy arrays are
list<double>, and precursor_mz/base_peak_intensity are double.

Training contains 275,810 distinct inchikey14 labels, exactly covering the
imported scoring-safe split's structure universe. Non-key source metadata
has missing values: instrument_type (33,053), precursor_error_ppm (4,286),
base_peak_intensity (1,163,138), collision_energy_ev (337,443), and
collision_energy_orig (320,464). These are preserved, not imputed. The
required IDs, labels and peak arrays have no nulls. Collision-energy lists
can have 1 to 6 entries, or be null, supporting the decision not to silently
reduce them to scalars.

Default preprocessing/binning was compared bitwise with the original CNN
notebook on all 1,213 test spectra and four rows per training batch (1,016
spectra across 254 batches). All comparisons passed. This is sampled
training preprocessing validation, not a transform of every raw spectrum.
Full-train chemical parsing, model retraining and formal validation were
not performed.

The actual local prototype Parquets were inspected directly: 17,456 training
spectra and 2,000 validation spectra. They retain 8 columns (spectral arrays,
precursor, adduct, normalized_smiles, inchikey14, ionization_mode, ingest_lib),
without native spectrum/query IDs or collision energy. They do not establish
the full raw train/test schema.

## API

```python
from casmi.data import load_train, load_test, iter_spectra

train = load_train(path="/path/to/train.parquet")
test = load_test(path="/path/to/test.parquet")
labels = load_train(columns=["smiles", "inchikey14"], path="/path/to/train.parquet")
for batch in iter_spectra("train", path="/path/to/train.parquet", batch_size=10000):
    ...
```

The reader does not download data. Default resolution is
`load_paths()["raw_dir"]/<split>.parquet`. With `CASMI_DATA_DIR`, both
`<root>/<split>.parquet` and `<root>/raw/<split>.parquet` are supported;
if both exist, pass an explicit path. Column projection returns requested
shared names only and avoids reading unrelated peak arrays. Streaming is
recommended for the 2.5-million-row training table.

Default loading preserves raw extras and adds shared aliases. Labels are
dropped from test outputs. Missing adduct or collision_energy is nullable;
missing peak/precursor fields is an error. Collision energy lists are not
silently averaged or converted to a scalar. Consumers needing a scalar must
choose an explicit policy; the scalar annotation on SpectrumRecord has not
been changed by this work.

## IDs and validation

- Native spectrum_id and query_id are preserved as strings.
- Test molecule_id is used as query_id when no query_id column exists.
- Missing test IDs raise an error; no test IDs are fabricated.
- When train has no IDs, `train:row:<physical-ordinal>` identifies each
  spectrum. Its default query_id is that same per-spectrum ID, explicitly
  marked `official_query_mapping=False` in DataFrame attrs. Equal labels
  are never grouped into a query.
- These generated ordinals refer to the exact input file's physical row
  order. Do not join ordinals across subsets, rewritten or differently
  ordered files. They are not official validation queries.
- Supply a DataFrame `query_map` with unique `spectrum_id, query_id`
  to provide an explicit upstream mapping. Missing mapped rows fail.
- Query mappings may be many spectra to one query. Duplicate spectrum IDs,
  invalid precursor values, mismatched arrays and nonfinite/negative peaks
  fail schema validation. Nullable structure labels are retained for the
  separate explicit `filter_labels` step.

Inspect live metadata after downloading the actual competition files:

```bash
PYTHONPATH=src python -m casmi.data.audit /path/to/train.parquet /path/to/test.parquet
```

This reports actual paths, row counts, row groups, Arrow fields and native IDs.
No raw data is committed, no frozen split is created, and no team contract is changed.
