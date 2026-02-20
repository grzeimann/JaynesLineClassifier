import numpy as np
from jlc.engine.flux_grid import FluxGrid


def test_ensure_threshold_expands_when_needed():
    fg = FluxGrid(Fmin=1e-18, Fmax=1e-16, n=64)
    thr = 1e-15
    changed = fg.ensure_threshold(thr)
    # Expect expansion since threshold is above initial Fmax
    assert changed is True
    assert fg.Fmin <= thr * 1e-2 + 1e-22  # THRESH_FACTOR_LOW and floor guard
    assert fg.Fmax >= thr * 1e2  # THRESH_FACTOR_HIGH
    # Grid should be rebuilt with n points
    F, logw = fg.grid(None)
    assert np.isfinite(F).all() and np.isfinite(logw).all()
    assert np.all(F > 0)
    assert F.size == fg.n


def test_ensure_threshold_no_change_if_already_covers():
    # Construct grid that already covers the threshold generously
    thr = 1e-16
    fg = FluxGrid(Fmin=thr * 1e-3, Fmax=thr * 1e3, n=64)
    changed = fg.ensure_threshold(thr)
    assert changed is False
    # Bounds remain the same
    assert np.isclose(fg.Fmin, thr * 1e-3)
    assert np.isclose(fg.Fmax, thr * 1e3)
