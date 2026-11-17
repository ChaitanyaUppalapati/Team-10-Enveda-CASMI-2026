"""Observed raw-schema fixtures and public reader contracts. Owner: Member 2."""

import numpy as np
import pandas as pd
import pytest

from casmi.data import check_spectra, iter_spectra, load_test, load_train, normalize_spectra
from casmi.data.audit import inspect_parquet
from casmi.interfaces import SPECTRUM_COLUMNS, SchemaError


def training_frame():
    return pd.DataFrame(
        {
            "ms2_mzs": [[100.1, 200.2], [50.0], [70.0]],
            "ms2_normalized_intensities": [[0.4, 1.0], [0.2], [0.5]],
            "precursor_mz": [300.0, 200.0, 150.0],
            "adduct": ["[M+H]+"] * 3,
            "normalized_smiles": ["CCO", "CCO", "CCN"],
            "inchikey14": ["A" * 14] * 2 + ["B" * 14],
            "collision_energy_ev": [[10.0, 20.0], [15.0], []],
            "ingest_lib": ["fixture"] * 3,
        }
    )


def test_train_maps_columns_without_label_based_queries(tmp_path):
    path = tmp_path / "train.parquet"
    training_frame().to_parquet(path, index=False)
    loaded = load_train(path=path)
    assert set(SPECTRUM_COLUMNS) <= set(loaded)
    assert list(loaded.query_id) == ["train:row:0", "train:row:1", "train:row:2"]
    assert loaded.query_id.nunique() == 3  # Two equal labels are not pooled into one query.
    assert loaded.attrs["official_query_mapping"] is False
    assert loaded.smiles.tolist() == ["CCO", "CCO", "CCN"]
    np.testing.assert_array_equal(loaded.collision_energy.iloc[0], [10.0, 20.0])
    assert check_spectra(loaded) is loaded


def test_streaming_ids_stable_under_batch_and_column_projection(tmp_path):
    path = tmp_path / "train.parquet"
    training_frame().to_parquet(path, index=False)
    a = pd.concat(iter_spectra("train", path=path, batch_size=1), ignore_index=True)
    b = pd.concat(iter_spectra("train", path=path, batch_size=2), ignore_index=True)
    assert a.spectrum_id.tolist() == b.spectrum_id.tolist()
    projected = load_train(["query_id", "smiles"], path=path)
    assert list(projected.columns) == ["query_id", "smiles"]
    assert projected.query_id.tolist() == a.query_id.tolist()
    keys_only = load_train(["inchikey14"], path=path)
    assert list(keys_only.columns) == ["inchikey14"] and len(keys_only) == 3


def test_test_uses_native_molecule_id_and_drops_labels(tmp_path):
    raw = training_frame().assign(molecule_id=["001", "001", "002"], spectrum_id=["s0", "s1", "s2"])
    path = tmp_path / "test.parquet"
    raw.to_parquet(path, index=False)
    loaded = load_test(path=path)
    assert loaded.query_id.tolist() == ["001", "001", "002"]
    assert loaded.spectrum_id.tolist() == ["s0", "s1", "s2"]
    assert not {"smiles", "normalized_smiles", "inchikey14"} & set(loaded)
    assert loaded.attrs["query_id_source"] == "native_molecule_id"


def test_explicit_query_map(tmp_path):
    path = tmp_path / "train.parquet"
    training_frame().to_parquet(path, index=False)
    mapping = pd.DataFrame({"spectrum_id": [f"train:row:{i}" for i in range(3)], "query_id": ["q", "q", "r"]})
    loaded = load_train(path=path, query_map=mapping)
    assert loaded.query_id.tolist() == ["q", "q", "r"]
    with pytest.raises(SchemaError, match="null/empty"):
        load_train(path=path, query_map=mapping.iloc[:2])
    with pytest.raises(SchemaError, match="exactly one"):
        load_train(path=path, query_map=pd.concat([mapping, mapping.iloc[:1]]))


def test_data_root_override_honored(tmp_path, monkeypatch):
    root = tmp_path / "external"
    (root / "raw").mkdir(parents=True)
    training_frame().to_parquet(root / "raw/train.parquet", index=False)
    monkeypatch.setenv("CASMI_DATA_DIR", str(root))
    assert len(load_train()) == 3
    training_frame().to_parquet(root / "train.parquet", index=False)
    with pytest.raises(ValueError, match="Multiple"):
        load_train()


def test_missing_file_and_wrong_dataset_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_train(path=tmp_path / "missing.parquet")
    path = tmp_path / "text.parquet"
    pd.DataFrame({"text": ["unrelated"], "label": [1]}).to_parquet(path, index=False)
    with pytest.raises(SchemaError, match="Missing spectrum columns"):
        load_train(path=path)


def test_test_never_fabricates_query_or_spectrum_ids():
    with pytest.raises(SchemaError, match="native spectrum_id"):
        normalize_spectra(training_frame(), split="test")
    with pytest.raises(SchemaError, match="native query_id/molecule_id"):
        normalize_spectra(training_frame().assign(spectrum_id=["s1", "s2", "s3"]), split="test")


def test_duplicate_ids_across_chunks_fail(tmp_path):
    path = tmp_path / "train.parquet"
    training_frame().assign(spectrum_id=["s1", "s2", "s1"]).to_parquet(path, index=False)
    with pytest.raises(SchemaError, match="across batches"):
        list(iter_spectra("train", path=path, batch_size=1))


def test_schema_validation_fails_bad_arrays_and_precursor():
    frame = normalize_spectra(training_frame(), split="train")
    frame.at[0, "mzs"] = np.array([100.0])
    with pytest.raises(SchemaError, match="Unequal/non-vector"):
        check_spectra(frame)
    frame = normalize_spectra(training_frame(), split="train")
    frame.at[0, "precursor_mz"] = np.nan
    with pytest.raises(SchemaError, match="finite and positive"):
        check_spectra(frame)


def test_metadata_audit_is_actual_file(tmp_path):
    path = tmp_path / "train.parquet"
    training_frame().to_parquet(path, index=False)
    audit = inspect_parquet(path)
    assert audit["rows"] == 3 and audit["native_query_id"] is None
    assert {field["name"] for field in audit["fields"]} == set(training_frame())
