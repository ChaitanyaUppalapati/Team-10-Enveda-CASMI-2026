# Public spectrum preprocessing

**Owner:** Member 2.

## Default behavior and CNN compatibility

`PreprocessConfig()` and `PreprocessConfig.cnn_compatible()` remove only
nonfinite/negative peaks. They preserve valid peak values and order, retain
all peaks (including those above precursor), perform no deisotoping, and
perform no square root. They keep every spectrum row, ID and label. Optional
label filtering is a separate explicit call.

The scaffold's previous defaults (1% threshold, 150 peaks, precursor trimming,
deisotoping and sqrt) had no implementation. They are not enabled implicitly
now because that would change inputs expected by the existing CNN weights.

```python
from casmi.preprocess import PreprocessConfig, preprocess_spectra, bin_spectrum

clean = preprocess_spectra(spectra, PreprocessConfig.cnn_compatible())
vector = bin_spectrum(
    clean.iloc[0]["mzs"], clean.iloc[0]["intensities"],
    max_mz=2200, bin_width=1.0, sqrt_intensity=True,
)
```

Both CNNs now delegate their binning to this shared function. It casts
intensities to float32, takes square root once, and maximum-pools per bin,
matching the original notebook. The public bin_spectrum default remains
max_mz=1000, bin_width=0.1, sqrt_intensity=False for other callers;
CNN callers explicitly choose 2200/1.0/True.

preprocess_spectra stores its settings in DataFrame attrs. A second requested
square root is rejected, and CNN datasets reject frames already marked as
sqrt-transformed. Pandas Parquet preserves these attrs; callers stripping
metadata must still respect the raw-intensity contract.

## Explicit optional operations

`min_rel_intensity` thresholds against the largest retained raw intensity.
`max_peaks` selects the strongest peaks with deterministic tie handling.
`remove_above_precursor` requires a finite positive precursor value.
`sqrt_intensity` is applied last. For CNN input, leave sqrt off in the
cleanup pipeline and let binning apply it.

The optional `deisotope` is a simple heuristic, not a validated chemistry
pipeline: it uses user-selected fixed charge, C13 spacing, a minimum of
three consecutive non-increasing positive peaks, and an explicit absolute
m/z tolerance. It sums an accepted envelope at the original monoisotopic m/z,
does not infer charges or recharge fragments, and keeps unmatched peaks.
It is disabled by default and has only synthetic unit checks, not a CASMI
performance benchmark. It is not an OpenMS implementation.

[OpenMS documentation](https://openms.org/documentation/html/classOpenMS_1_1Deisotoper.html)
describes the dependence on isotope spacing, charge, resolution and intensity
assumptions and notes limitations outside peptide fragments. This informs
the choice to keep any heuristic opt-in; it does not validate our heuristic.

`filter_labels` validates presence/parseability of SMILES and the syntax of
14-letter scoring keys. It does not canonicalize SMILES, rewrite labels, or
prove SMILES/key identity; scoring-key audit and split ownership remain
separate. It returns a filtered copy with input/kept/dropped row counts.

The cleanup pipeline updates canonical peaks and retained raw-name aliases
together. Source metadata such as num_peaks/base_peak_intensity is preserved
as source metadata, not recomputed statistics of transformed peaks.
