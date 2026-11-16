# data/

**Owner:** Member 2 (`data/splits/` belongs to Member 5).

Everything here except this README and `splits/` is gitignored. Never commit competition data.

```
data/
  raw/          competition train/test parquet (download from Kaggle)
  interim/      scratch outputs
  processed/    preprocessed spectra, fingerprints (casmi.fingerprints), binned arrays
  candidates/   COCONUT / PubChem tables (Member 4)
  splits/       FROZEN validation split, committed (Member 5)
```

Download (needs the Kaggle CLI and an accepted competition rules page):

```bash
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data/raw
```

To keep data outside the repo, set `CASMI_DATA_DIR`.
