import numpy as np
import pytest

from jlc.selection.base import SNLogisticPerLambdaBin, SelectionModel
from jlc.selection import build_selection_model_from_priors
from jlc.priors import PriorRecord


def test_sn_logistic_per_lambda_bin_basic_properties():
    bins = np.array([3600.0, 4500.0, 5400.0])  # 2 bins
    sn50 = np.array([6.0, 6.0])
    width = np.array([1.0, 1.0])
    model = SNLogisticPerLambdaBin(bins, sn50, width)

    # Below sn50 -> < 0.5; at sn50 ~ 0.5; above sn50 -> > 0.5
    c_low = model.completeness(4.0, 4000.0, {})
    c_mid = model.completeness(6.0, 4000.0, {})
    c_high = model.completeness(8.0, 4000.0, {})
    assert 0.0 <= c_low < 0.5
    assert np.isclose(c_mid, 0.5, atol=1e-6)
    assert 0.5 < c_high <= 1.0

    # Monotonic with S/N in a given bin
    sns = np.linspace(0.0, 12.0, 25)
    cs = [model.completeness(s, 5000.0, {}) for s in sns]
    assert all(cs[i] <= cs[i+1] + 1e-12 for i in range(len(cs)-1))

    # Clamp behavior at extremes
    assert 0.0 <= model.completeness(-1e9, 4000.0, {}) <= 1.0
    assert 0.0 <= model.completeness(+1e9, 5200.0, {}) <= 1.0


def test_build_selection_model_from_priors_happy_path():
    record = PriorRecord(
        name="lae_sel",
        scope="label",
        label="lae",
        hyperparams={
            "selection": {
                "default_sigma": 5.0e-18,
                "sn": {
                    "model": "logistic_per_lambda_bin",
                    "params": {
                        "bins_wave": [3600, 4500, 5400],
                        "sn50": [6.0, 6.0],
                        "width": [1.0, 1.0],
                    },
                },
            }
        },
    )
    sel = build_selection_model_from_priors(record)
    assert isinstance(sel, SelectionModel)
    # Noise attached
    assert sel.noise_model is not None
    # SN model bound to label
    snm = sel.sn_model_for_label("lae")
    assert snm is not None


def test_build_selection_model_from_priors_missing_params_returns_none():
    # Missing width
    record = PriorRecord(
        name="bad_sel",
        scope="label",
        label="lae",
        hyperparams={
            "selection": {
                "default_sigma": 5.0e-18,
                "sn": {
                    "model": "logistic_per_lambda_bin",
                    "params": {
                        "bins_wave": [3600, 4500, 5400],
                        "sn50": [6.0, 6.0],
                    },
                },
            }
        },
    )
    sel = build_selection_model_from_priors(record)
    assert sel is None
