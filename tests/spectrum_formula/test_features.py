"""Tests for the spectrum + formula models' numpy-only code. Owner: Member 4."""

import importlib.util

import numpy as np
import pytest
from sf_common import data as D
from sf_common import features as F

from casmi.candidates.lookup import ADDUCT_DELTAS
from casmi.preprocess.binning import bin_spectrum


# ---------- binning ----------
def _flat(spectra):
    mzs = np.concatenate([s[0] for s in spectra])
    ints = np.concatenate([s[1] for s in spectra])
    offsets = np.concatenate([[0], np.cumsum([len(s[0]) for s in spectra])])
    return mzs, ints, offsets


def test_vectorised_binning_matches_member2_bin_spectrum():
    rng = np.random.default_rng(0)
    spectra = [(np.sort(rng.uniform(50, 900, n)), rng.uniform(0, 1, n)) for n in (5, 40, 1, 17)]
    mzs, ints, offsets = _flat(spectra)
    got = F.bin_flat_peaks(mzs, ints, offsets, precursor_mz=np.full(4, 1000.0))
    for row, (m, i) in zip(got, spectra, strict=True):
        expected = bin_spectrum(m, i, max_mz=F.MAX_MZ, bin_width=F.BIN_WIDTH, sqrt_intensity=True)
        np.testing.assert_allclose(row, expected, rtol=1e-6)


def test_finer_bin_width_matches_member2_bin_spectrum():
    rng = np.random.default_rng(1)
    spectra = [(np.sort(rng.uniform(50, 900, n)), rng.uniform(0, 1, n)) for n in (12, 30)]
    mzs, ints, offsets = _flat(spectra)
    got = F.bin_flat_peaks(mzs, ints, offsets, precursor_mz=np.full(2, 1000.0), bin_width=0.25)
    assert got.shape == (2, F.n_bins(0.25)) == (2, 4800)
    for row, (m, i) in zip(got, spectra, strict=True):
        np.testing.assert_allclose(
            row, bin_spectrum(m, i, max_mz=F.MAX_MZ, bin_width=0.25, sqrt_intensity=True), rtol=1e-6
        )


def test_neutral_loss_block_matches_member2_bin_spectrum_of_the_losses():
    rng = np.random.default_rng(2)
    precursor = np.array([700.0, 450.0])
    spectra = [
        (np.sort(rng.uniform(50, p - 5, n)), rng.uniform(0, 1, n)) for n, p in zip((15, 8), precursor, strict=True)
    ]
    mzs, ints, offsets = _flat(spectra)
    got = F.bin_flat_peaks(mzs, ints, offsets, precursor_mz=precursor, bin_width=0.5, losses=True)
    nb = F.n_bins(0.5)
    assert got.shape == (2, F.input_dim(0.5, True)) == (2, 2 * nb)
    for row, (m, i), p in zip(got, spectra, precursor, strict=True):
        expected = bin_spectrum(p - m, i, max_mz=F.MAX_MZ, bin_width=0.5, sqrt_intensity=True)
        np.testing.assert_allclose(row[nb:], expected, rtol=1e-6)
        mz_block = bin_spectrum(m, i, max_mz=F.MAX_MZ, bin_width=0.5, sqrt_intensity=True)
        np.testing.assert_allclose(row[:nb], mz_block, rtol=1e-6)


def test_spec_args_defaults_and_overrides():
    assert F.spec_args({}) == {"bin_width": 1.0, "losses": False}
    assert F.spec_args({"bin_width": 0.5, "neutral_loss": True}) == {"bin_width": 0.5, "losses": True}


def test_binning_drops_peaks_above_precursor_plus_margin():
    mzs, ints, offsets = _flat([(np.array([100.0, 301.9, 302.5]), np.array([1.0, 1.0, 1.0]))])
    row = F.bin_flat_peaks(mzs, ints, offsets, precursor_mz=np.array([300.0]))[0]
    assert row[100] == 1.0 and row[301] == 1.0 and row[302] == 0.0


def test_binning_handles_empty_spectra_and_max_pools():
    mzs, ints, offsets = _flat([(np.array([]), np.array([])), (np.array([10.2, 10.7]), np.array([0.25, 1.0]))])
    out = F.bin_flat_peaks(mzs, ints, offsets, precursor_mz=np.array([500.0, 500.0]))
    assert out[0].sum() == 0 and out[1][10] == 1.0 and out.shape == (2, F.N_BINS)


# ---------- meta features ----------
def test_meta_features_shape_adduct_and_polarity():
    mz = [195.087652, 193.073, 300.0]
    m = F.meta_features(mz, ["[M+H]+", "[M-H]-", "[weird]+"], [20.0, float("nan"), 10.0])
    assert m.shape == (3, F.N_META)
    onehot = m[:, 2 : 2 + len(F.ADDUCTS) + 1]
    assert onehot.sum(axis=1).tolist() == [1, 1, 1]
    assert onehot[0, F.ADDUCTS.index("[M+H]+")] == 1 and onehot[2, -1] == 1  # unknown -> "other"
    assert m[:, -3].tolist() == [1.0, 0.0, 1.0]  # polarity comes from the trailing charge sign


def test_meta_features_collision_energy_missing_flag():
    m = F.meta_features([200.0, 200.0], ["[M+H]+", "[M+H]+"], [float("nan"), 35.0])
    assert m[:, -1].tolist() == [1.0, 0.0] and m[0, -2] == 0.0 and m[1, -2] > 0


def test_meta_features_neutral_mass_uses_adduct():
    m = F.meta_features([195.087652], ["[M+H]+"], [20.0])
    assert m[0, 1] * F.MAX_MZ == pytest.approx(194.0804, abs=1e-3)


def test_mean_collision_energy():
    assert F.mean_collision_energy([10.0, 20.0]) == 15.0
    assert F.mean_collision_energy(25.0) == 25.0
    assert np.isnan(F.mean_collision_energy(None)) and np.isnan(F.mean_collision_energy([]))
    assert F.mean_collision_energy([float("nan"), 30.0]) == 30.0


def test_all_test_adducts_have_a_dedicated_slot():
    assert set(F.ADDUCTS) == set(ADDUCT_DELTAS)


# ---------- formula ----------
def test_formula_counts():
    counts = dict(zip(F.ELEMENTS, F.formula_counts("C8H10N4O2"), strict=True))
    assert (counts["C"], counts["H"], counts["N"], counts["O"], counts["S"]) == (8, 10, 4, 2, 0)
    assert dict(zip(F.ELEMENTS, F.formula_counts("C6H5ClBrNaO"), strict=True))["Cl"] == 1  # Na ignored, Cl not "C"+"l"
    assert F.formula_counts("").sum() == 0
    np.testing.assert_allclose(F.log_counts(np.array([0.0, 1.0])), [0.0, np.log(2)])


# ---------- torch / rdkit gated ----------
def test_model_forward_shapes():
    torch = pytest.importorskip("torch")
    from sf_common.model import DualBranchNet

    net = DualBranchNet(n_bits=64, hidden=32, spec_dim=16, meta_dim=8)
    fp, formula = net(torch.zeros(3, F.N_BINS), torch.zeros(3, F.N_META))
    assert fp.shape == (3, 64) and formula.shape == (3, len(F.ELEMENTS))


@pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="requires the chem extra (rdkit)")
def test_morgan_bits_shape_and_failure_row():
    bits = D.morgan_bits(["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "not smiles"], n_bits=256, workers=1)
    assert bits.shape == (2, 256) and bits[0].sum() > 0 and bits[1].sum() == 0


def test_priors_modes_and_alignment():
    import pandas as pd
    from sf_common import priors

    table = pd.DataFrame(
        {
            "FirstBlock": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
            "PubMed_Count": [0, 9],
            "Patent_Count": [0, 90],
            "AnnoTypeCount": [1, 4],
        }
    )
    keys = ["BBBBBBBBBBBBBB", "ZZZZZZZZZZZZZZ", "AAAAAAAAAAAAAA"]
    assert priors.pool_prior(keys, table, "member").tolist() == [1.0, 0.0, 1.0]
    assert priors.pool_prior(keys, table, "pubmed").tolist() == pytest.approx([np.log1p(9), 0.0, 0.0])
    assert priors.pool_prior(keys, table, "both").tolist() == pytest.approx([np.log1p(99), 0.0, 0.0])
    assert priors.pool_prior(keys, table, "annotations").tolist() == [4.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        priors.prior_values(table, "nope")


def test_priors_config_and_counts_parquet_roundtrip(tmp_path):
    import pandas as pd
    from sf_common import priors

    csv = tmp_path / "pcl.csv"
    pd.DataFrame(
        {
            "FirstBlock": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
            "PubMed_Count": [0, 9],
            "Patent_Count": [3, 90],
            "AnnoTypeCount": [1, 4],
            "Other": ["x", "y"],
        }
    ).to_csv(csv, index=False)
    priors.save_counts(csv, tmp_path / "pubchemlite_counts.parquet")
    keys = ["BBBBBBBBBBBBBB", "AAAAAAAAAAAAAA"]
    prior, kwargs = priors.config_prior(keys, {"mode": "pubmed", "weight": 0.8, "topk": 25}, tmp_path)
    assert prior.tolist() == pytest.approx([np.log1p(9), 0.0]) and kwargs == {"prior_weight": 0.8, "prior_topk": 25}
    assert priors.config_prior(keys, {"weight": 0.0}, tmp_path) == (None, {})  # off
    assert priors.config_prior(keys, None, tmp_path) == (None, {})
    with pytest.raises(FileNotFoundError):
        priors.config_prior(keys, {"weight": 1.0, "table": "missing.parquet"}, tmp_path)
