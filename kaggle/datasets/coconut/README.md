# kaggle/datasets/coconut/

**Owner:** Member 4. Packages a COCONUT snapshot as the candidate pool for the offline notebook.
`build_dataset.py` writes `build/coconut.parquet` with the columns
`inchikey14, smiles, formula, monoisotopic_mass`.

Build and audit (both use all cores; the build takes about 80 s, the audit about 35 s):

```bash
python kaggle/datasets/coconut/build_dataset.py --source <coconut_csv_lite.csv> --out data/candidates
CASMI_CANDIDATES_DIR=data/candidates python kaggle/datasets/coconut/coverage_audit.py
```

Source: `coconut_csv_lite-11-2026.zip` from <https://coconut.naturalproducts.net/download>
(`https://coconut.s3.uni-jena.de/prod/downloads/2026-11/coconut_csv_lite-11-2026.zip`, 199 MB,
CC0, sha256 `81f3edc1731f01da289fe6b84d162667863e331d52041a781565e1af2eb79939`).
Standardisation: largest fragment, RDKit 2026.03.3 InChIKey, first 14 characters, neutral mass 100-1500 Da.
738,793 unique SMILES become 471,809 unique `inchikey14` (stereoisomers collapse; about 1% fall outside the mass range).

| snapshot | n structures | val coverage | notes |
|----------|--------------|--------------|-------|
| COCONUT 2026-11 (lite) | 471,809 | **provisional**: 0.980 on `enveda-np-examples` (250 structures) at 10 ppm; 0.517 on all libraries pooled | Proxy truth from train, not the frozen split (Member 5). Re-run when the split lands. |

Provisional coverage at 10 ppm by training library (true structure among the mass-matched candidates):

| library | structures | in pool | coverage @5 ppm | @10 ppm | @20 ppm | median / p90 candidates @5 ppm |
|---|---|---|---|---|---|---|
| enveda-np-examples (closest to test) | 250 | 0.996 | 0.964 | 0.980 | 0.980 | 37 / 144 |
| msdial | 2000 | 0.836 | 0.823 | 0.823 | 0.823 | 25 / 152 |
| mona | 2000 | 0.820 | 0.786 | 0.802 | 0.812 | 25 / 155 |
| riken | 2000 | 0.804 | 0.730 | 0.776 | 0.794 | 23 / 138 |
| masaryk | 416 | 0.837 | 0.812 | 0.815 | 0.817 | 9 / 52 |
| drug_plus | 2000 | 0.755 | 0.724 | 0.725 | 0.725 | 11 / 73 |
| massbank | 2000 | 0.645 | 0.608 | 0.626 | 0.632 | 12 / 114 |
| gnps | 2000 | 0.376 | 0.356 | 0.368 | 0.375 | 13 / 82 |
| spectraverse | 2000 | 0.241 | 0.236 | 0.240 | 0.240 | 9 / 97 |
| pluskal_ms2 | 2000 | 0.173 | 0.172 | 0.172 | 0.172 | 13 / 67 |
| enveda-180 (synthetic) | 2000 | 0.000 | 0.000 | 0.000 | 0.000 | 12 / 54 |

Caveats: low rows are expected (synthetic or non-COCONUT chemistry). The np-examples number is likely
optimistic for the hidden test, which also contains structures outside COCONUT. Spectra were limited to
the 10 test adducts with |precursor error| <= 20 ppm.
