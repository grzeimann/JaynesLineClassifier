import numpy as np
from jlc.simulate.rate_integrand import rate_density_integrand_per_flux
from jlc.population.schechter import SchechterLF


class DummyProvider:
    def __init__(self, c=1.0):
        self.c = float(c)
    def completeness(self, F_true_grid, lam, label):
        F = np.asarray(F_true_grid, dtype=float)
        return np.ones_like(F, dtype=float) * self.c


def test_rate_integrand_uses_schechter_and_completeness():
    F = np.logspace(-18, -14, 64)
    lf = SchechterLF(log10_Lstar=42.0, alpha=-1.5, log10_phistar=-3.0, Lmin=1e38, Lmax=1e46)
    # With completeness 1.0
    r1 = rate_density_integrand_per_flux(F, 6000.0, "lae", lf, DummyProvider(1.0))
    # With completeness 0.5 it should scale linearly down
    r2 = rate_density_integrand_per_flux(F, 6000.0, "lae", lf, DummyProvider(0.5))
    assert r1.shape == F.shape
    assert np.all(r1 >= 0)
    assert np.allclose(r2, 0.5 * r1)


def test_rate_integrand_handles_bad_inputs_gracefully():
    F = np.array([0.0, -1.0, np.nan, 1e-17])
    lf = SchechterLF(log10_Lstar=42.0, alpha=-1.5, log10_phistar=-3.0)
    r = rate_density_integrand_per_flux(F, 5000.0, "oii", lf, DummyProvider(1.0))
    # Non-finite or non-positive inputs should yield finite, non-negative outputs
    assert r.shape == F.shape
    assert np.all(np.isfinite(r) | np.isnan(F))
    assert np.all(r[np.isfinite(r)] >= 0)
