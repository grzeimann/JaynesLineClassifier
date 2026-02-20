import numpy as np
from jlc.selection.base import SelectionModel


def test_selection_hard_threshold_basic():
    sel = SelectionModel(f_lim=1.0)
    F = np.array([0.5, 1.0, 2.0])
    C = sel.completeness(F, wave_obs=7000.0)
    assert np.all(C >= 0.0) and np.all(C <= 1.0)
    assert C[0] == 0.0
    assert C[1] == 0.0  # strictly greater-than threshold per implementation
    assert C[2] == 1.0


def test_selection_smooth_tanh_monotonic():
    sel = SelectionModel(F50=1.0, w=0.5)
    F = np.array([0.01, 0.5, 1.0, 2.0, 10.0])
    C = sel.completeness(F, wave_obs=7000.0)
    assert np.all(C[:-1] <= C[1:] + 1e-12)  # non-decreasing
    assert np.all((C >= 0.0) & (C <= 1.0))
    # symmetry-ish: at F=F50 completeness ~ 0.5
    assert np.isclose(C[2], 0.5, rtol=0, atol=1e-2)


def test_ra_dec_factor_modulation_and_clamp():
    # Factor returns >1 or <0 should be clamped to [0,1]
    def g(ra, dec, lam):
        # make it alternate a bit based on lam to avoid constant folding
        return 1.5 if lam > 7000 else -0.2

    sel = SelectionModel(F50=1.0, w=0.1, ra_dec_factor=g)
    F = np.array([10.0])  # well above threshold -> base completeness ~1
    C1 = sel.completeness(F, wave_obs=8000.0, ra=0.0, dec=0.0)
    C2 = sel.completeness(F, wave_obs=6000.0, ra=0.0, dec=0.0)
    # First case should clamp to 1.0, second to 0.0
    assert np.isclose(C1[0], 1.0)
    assert np.isclose(C2[0], 0.0)
