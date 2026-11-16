# kaggle/datasets/casmi-code/

**Owner:** Member 6. Packages the code for the offline notebook:

```bash
python kaggle/datasets/casmi-code/build_dataset.py --weights outputs/weights/<model_id>
kaggle datasets version -p kaggle/datasets/casmi-code/build -m "weekly update"
```

The first upload uses `kaggle datasets create -p kaggle/datasets/casmi-code/build` instead.
Set your Kaggle username in `dataset-metadata.json` first.
