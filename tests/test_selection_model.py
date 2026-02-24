import numpy as np
from jlc.selection.base import SelectionModel, NoiseModel, SNLogisticPerLambdaBin


def test_selection_sn_array_neutral_when_unconfigured():
    # Without noise/SN models, completeness should be neutral (ones)
    sel = SelectionModel()
    F = np.array([0.5, 1.0, 2.0])
    C = sel.completeness_sn_array("all", F, wave_true=4000.0)
    assert np.allclose(C, np.ones_like(F))


def test_selection_sn_array_with_logistic_model():
    # Configure a simple SN logistic model with constant sigma
    nm = NoiseModel(default_sigma=1.0)
    bins = np.array([3000.0, 5000.0])  # single bin
    sn50 = np.array([1.0])
    width = np.array([0.5])
    snm = SNLogisticPerLambdaBin(bins, sn50, width)
    sel = SelectionModel(noise_model=nm, sn_models={"lae": snm})
    F = np.array([0.01, 0.5, 1.0, 2.0, 10.0])  # with sigma=1, these are S/N
    C = sel.completeness_sn_array("lae", F, wave_true=4000.0)
    # Non-decreasing and clamped
    assert np.all(C[:-1] <= C[1:] + 1e-12)
    assert np.all((C >= 0.0) & (C <= 1.0))
    # At S/N=sn50 completeness ~ 0.5
    mid_idx = int(np.where(np.isclose(F, 1.0))[0][0])
    assert np.isclose(C[mid_idx], 0.5, atol=5e-2)
