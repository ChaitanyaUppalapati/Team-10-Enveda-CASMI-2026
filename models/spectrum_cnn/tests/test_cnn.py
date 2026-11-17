"""Member 2 regression checks: checkpoint compatibility, inputs and candidate isolation."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="CNN tests require the optional ml dependencies")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.architectures import build_model  # noqa: E402
from common.data import check_split, spectrum_to_bins  # noqa: E402
from common.runner import output_location, predict, score_candidates  # noqa: E402


@pytest.fixture
def model_cfg():
    return {"architecture": "cnn1d", "fp_size": 1024, "fp_radius": 2, "mz_max": 2200, "bin_width": 1.0}


def test_binning_filters_invalid_and_max_pools():
    vector = spectrum_to_bins([1.1, 1.9, -1, 2200, 2, np.nan], [4, 9, 9, 9, -1, 9])
    assert len(vector) == 2200
    assert vector[1] == 3 and np.count_nonzero(vector) == 1
    with pytest.raises(ValueError, match="equally sized"):
        spectrum_to_bins([1], [])


@pytest.mark.parametrize("architecture", ["cnn1d", "multires_cnn"])
def test_models_accept_notebook_input(architecture):
    torch.set_num_threads(2)
    model = build_model(architecture, 1024).eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 1, 2200)).shape == (2, 1024)


def test_split_rejects_wrong_membership(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_structures": ["A"], "val_structures": ["B"]}))
    with pytest.raises(ValueError, match="do not belong"):
        check_split(pd.DataFrame({"inchikey14": ["B"]}), pd.DataFrame({"inchikey14": ["A"]}), split)


def test_candidate_scores_pool_only_real_queries(model_cfg):
    pytest.importorskip("rdkit", reason="Candidate fingerprints require the optional chem dependencies")
    # Values make the expected score analytically known, independent of chemistry implementation.
    from common.data import fingerprint

    bits = fingerprint("CCO").astype(np.float32)
    candidates = pd.DataFrame({"query_id": ["q1", "q2"], "inchikey14": ["A" * 14, "B" * 14], "smiles": ["CCO", "CCO"]})
    probs = np.stack([bits, np.zeros(1024), bits])
    scores = score_candidates(probs, ["q1", "q1", "q2"], candidates, model_cfg)
    np.testing.assert_allclose(scores.score, [0.5, 1.0], atol=1e-7)
    assert list(scores.query_id) == ["q1", "q2"]


@pytest.mark.parametrize("problem", ["duplicate", "missing_query", "null_smiles", "bad_key", "invalid_smiles"])
def test_bad_candidate_inputs_rejected(model_cfg, problem):
    if problem == "invalid_smiles":
        pytest.importorskip("rdkit", reason="SMILES validation requires the optional chem dependencies")
    frame = pd.DataFrame({"query_id": ["q1"], "inchikey14": ["A" * 14], "smiles": ["CCO"]})
    if problem == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    elif problem == "missing_query":
        frame.loc[0, "query_id"] = "unknown"
    elif problem == "null_smiles":
        frame.loc[0, "smiles"] = None
    elif problem == "bad_key":
        frame.loc[0, "inchikey14"] = "invalid"
    elif problem == "invalid_smiles":
        frame.loc[0, "smiles"] = "not a molecule"
    with pytest.raises(ValueError):
        score_candidates(np.zeros((1, 1024)), ["q1"], frame, model_cfg)


def test_no_automatic_oracle_or_formal_val(model_cfg):
    cfg = {"model": model_cfg}
    with pytest.raises(ValueError, match="--spectra and --candidates"):
        predict(cfg, "val", prototype=True)
    with pytest.raises(ValueError, match="frozen split"):
        predict(cfg, "val", spectra_path="placeholder", candidates_path="placeholder")


def test_prototype_output_is_separate(tmp_path):
    root, model_id = output_location({"model_id": "spectrum_cnn.arch1_cnn1d"}, True, tmp_path)
    assert root == tmp_path / "prototype"
    assert model_id.endswith(".prototype")
