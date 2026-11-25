# Competition summary

**Owner:** Member 7

Sources: the Kaggle competition pages (Overview, Data, Rules, Code Requirements) and the host's
forum clarification on external data, checked 2026-11-22; plus Member 2's audit of the downloaded
files (`src/casmi/data/README.md`).

## Key dates

This competition is our DATA 266 term project, so **the course deadline, not Kaggle's, is the one
we plan to**. Internal dates come from the team work schedule (`CASMI 2026 Team Work Schedule.xlsx`).

| Date | What |
|---|---|
| Nov 24 | Architecture 2 models due (val MRR@25 per class + leaderboard submission) |
| Nov 25 | Report draft v1; offline runtime test of the best pipeline |
| Nov 26 | Final model selection on the frozen holdout |
| **Nov 27** | **Final Kaggle submission + screenshot evidence** (team-set date, leaving buffer before Dec 1) |
| Nov 29 | Final report, slides, clean reproducible repo, presentation rehearsal |
| Nov 30 | Peer review forms |
| **Dec 1** | **Course project deadline** |

Kaggle's own dates (Dec 7 entry and team-merger deadline, Dec 14 final submission) fall after our
course deadline. The course grades the best leaderboard result **at the course cutoff**.

## Task

Enveda CASMI 2026 ("Molecule ID From Mass Spectra") is a Kaggle **code competition**. Given the
tandem mass spectrometry (MS/MS) spectra of an unknown small molecule, identify its **2D
structure**.

- Predictions are made **per molecule, not per spectrum**. A molecule may be measured several
  times (different collision energies or adducts), and all of its spectra must be merged into a
  **single ranked list**.
- For each `molecule_id`, submit up to **25 candidate structures as SMILES**, best guess first.
- Chemistry in scope: natural-product-like molecules from plants, mammals or microbes (confirmed
  and hypothesised natural products, analogs, and plausible synthetic look-alikes).

Why it matters: most metabolites in natural samples have no reference spectrum in any public
library, so they cannot be identified by library lookup. Naming them from the spectrum alone is
the bottleneck in natural-product drug discovery.

### Three novelty classes (hidden per molecule)

| Class | Definition | What it implies for us |
|---|---|---|
| 1. In public spectral libraries | A public reference MS/MS spectrum exists | Library search (B2) can find it |
| 2. Known structure, no public spectra | Structure is in PubChem or COCONUT, never measured | Needs a model that scores a structure without a reference spectrum (fingerprint prediction + retrieval) |
| 3. Novel structure | Not in PubChem at all | Not in any candidate pool; only de novo generation could hit it |

The class mix of the hidden test set is not disclosed.

## Data

| File | Contents |
|---|---|
| `train.parquet` | **2,539,608** spectra of **275,810** unique structures (`inchikey14`), one row per spectrum, structure given |
| `test.parquet` | Visible placeholder: **1,213** spectra of **400** molecules (1–9 spectra each). Built from training examples so code runs end to end; **replaced by the hidden test set at scoring**, so it cannot be used for tuning |
| `sample_submission.csv` | 400 rows, columns `molecule_id, smiles` |

**Hidden test set:** about 1,500 spectra from about 400 molecules, 1–16 spectra per molecule
(median 3), all acquired on a **Bruker timsTOF**, monoisotopic masses 157–1,159 Da (median 348),
none previously published.

**Key columns** (shared by train and test): `ms2_mzs` + `ms2_normalized_intensities` (aligned
peak arrays, base peak = 1.0), `precursor_mz`, `adduct`, `ionization_mode`, `instrument_type`,
`collision_energy_ev` (a list; several values mean merged acquisitions), `base_peak_intensity`.
Test adds `molecule_id`, `spectrum_id`. Train adds the label (`normalized_smiles`, `inchikey`,
`inchikey14`), `molecular_formula`, `ingest_lib`, `adduct_orig` and `precursor_error_ppm`
(uncleaned, so also a label-quality signal).

**Adducts in test (10):** positive `[M+H]+`, `[M+NH4]+`, `[M-H2O+H]+`, `[M-2H2O+H]+`, `[M+Na]+`,
`[M+K]+`; negative `[M-H]-`, `[M-H2O-H]-`, `[M+CH2O2-H]-`, `[M+Cl]-`. Train contains many more.

**Training libraries (`ingest_lib`):**

| Library | Spectra | Structures | Note |
|---|---:|---:|---|
| enveda-180 | 1,153,785 | 182,941 | Same timsTOF as test, but synthetic drug-like chemistry |
| pluskal_ms2 | 527,581 | 46,821 | Orbitrap, multiple collision energies |
| riken | 347,171 | 15,892 | Plant specialised metabolites |
| gnps | 220,849 | 45,750 | Largest natural-product collection, most heterogeneous |
| massbank | 101,727 | 9,180 | Curated, many labs/instruments |
| mona | 92,416 | 11,681 | MassBank of North America |
| spectraverse | 50,933 | 9,631 | Harmonised aggregation |
| msdial | 40,765 | 9,127 | MS-DIAL distributions |
| drug_plus | 2,545 | 2,539 | Drugs, ~1 spectrum each, no collision energy |
| enveda-np-examples | 1,151 | 250 | **Closest to test**: natural products, same instrument and pipeline |
| masaryk | 652 | 416 | Chemical standards |

Structures recur across libraries, so per-library counts sum to more than 275,810. Only
enveda-180 and enveda-np-examples are uniformly timsTOF like the test set.

**Data quirks to handle:** missing `collision_energy_ev` (337,443 train rows), missing
`instrument_type` (33,053), missing `base_peak_intensity` (1,163,138), collision-energy units in
train are eV / NCE / V / unknown (always eV in test).

## Metric (MRR@25, InChIKey14)

Mean reciprocal rank at 25, computed **per molecule**: MRR@25 = (1/N) Σ 1/rᵢ, where rᵢ is the
position of the **first** correct structure for molecule *i* (0 if not in the top 25).

- Rank 1 → 1.0, rank 2 → 0.5, rank 4 → 0.25, rank 25 → 0.04, missing → 0.
- **Matching:** both our SMILES and the answer go through RDKit tautomer canonicalisation (pinned
  **2026.03.3**) and are reduced to the **InChIKey14** (first InChIKey block = atom connectivity).
  Stereochemistry and tautomer form are not penalised: both glucose SMILES with and without
  stereocentres map to `WQZGKKKJIJFFOK`.
- Official scorer: [metric notebook](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank).
  Our local scorer (`src/casmi/evaluation/`, Member 5) should reproduce it exactly.

Consequences for us:
- Pin RDKit 2026.03.3 so local and leaderboard keys agree.
- Two guesses with the same InChIKey14 waste a slot; deduplicate before writing the top 25.
- A wrong guess costs nothing beyond the slot it occupies, so always fill all 25.
- A near-miss at rank 2–5 still earns real credit, so ranking quality matters, not only top-1.

### Submission file

`submission.csv`, header `molecule_id,smiles`, one row per `molecule_id`, up to 25 SMILES
**joined by semicolons in one field**, best first:

```
molecule_id,smiles
m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```

Rejected if a column is missing, the file is empty, any value is null, a `molecule_id` repeats, or a
molecule has more than 25 guesses (Member 6's checker, `src/casmi/submission/`).

## Rules

**Team and submissions**
- **We follow the course requirements**; Kaggle prize eligibility is not a goal.
- **Course requirement:** every member develops 2–3 architectures and submits their work to the
  leaderboard (DATA 266 project brief). Grading uses the team's submissions over time (G.4) and the
  team's best score and improvement over baseline at the course cutoff (G.6).
- **Kaggle mechanics:** a Kaggle team has at most **5** members and **5 submissions per day**, so
  plan submissions early; models are chosen by validation MRR@25, not the public leaderboard.

**Validation must be our own.** The visible test file is a placeholder, so the team uses a
**structure-level split**: 10% of unique `inchikey14` values go to validation, and **all** their
spectra are removed from the search library (e.g. if caffeine is held out, all its training
spectra are deleted and the pipeline must find caffeine without them). Owner: Member 5,
`data/splits/`. Hand-labelling or human prediction of test or validation records is prohibited.

**External data and models**
- Allowed if **freely and publicly available and equally accessible to all**, including pretrained
  models. Per the host's forum update, open data with **non-commercial** licences (e.g. CC BY-NC)
  is now allowed.
- **Not allowed:** proprietary or paid data, anything not freely available. **NIST is explicitly
  banned**, including models trained on it.
- Candidate sources we plan to use (COCONUT, PubChem, GNPS, MassBank, MoNA, DreaMS/GeMS,
  FragHub, matchms, SIRIUS) should each be checked against this rule before use.

**Data and code handling**
- Competition data is CC BY-NC 4.0, non-commercial, and **must not be redistributed** to anyone
  outside the competition. Never commit data to this repository.
- **No private code sharing outside the Kaggle team.** Public sharing is allowed only on Kaggle
  (forums or notebooks), under an open-source licence.
- Winners must release code under an open-source licence (MIT) and provide reproducible training and
  inference code.

### How the test spectra were cleaned

Useful for matching our training preprocessing to the test set:
- spectra whose precursor mass disagreed with the known structure and adduct were dropped;
- spectra with a base peak below 1,000 raw counts were dropped;
- peaks above precursor + 2 Da were removed.

Nothing else was done (no intensity floor, no deisotoping), so those choices are ours. The Data page
lists common options: a 0.1–2% relative-intensity floor, at least 5–6 peaks, top-N peaks (e.g. 128)
or top-6 per 50 Da, deisotoping, and square-root or log intensity.

## Offline 9-hour limit

- Submissions are made **only through Kaggle Notebooks**. The Submit button is active only if the
  notebook runs in **≤ 9 hours (CPU or GPU)** with **internet disabled** and writes `submission.csv`.
- At scoring, Kaggle swaps in the hidden test set and **re-runs the notebook on its own machine**;
  we never see the hidden spectra or download predictions.
- Everything the notebook needs (the `casmi` package, model weights, COCONUT/PubChem candidate
  tables) must be uploaded beforehand as Kaggle Datasets (`kaggle/`, owner Member 6; COCONUT:
  Member 4).
- Training can happen anywhere (local GPU, Colab, Kaggle); only inference is time-limited.
- Budget per stage is tracked in [`runtime_report.md`](runtime_report.md). Candidate retrieval and
  scoring over large databases is the likely bottleneck, so precomputed candidate fingerprints and
  mass indexes should be packaged rather than computed at inference time.
