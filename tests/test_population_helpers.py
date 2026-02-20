import numpy as np
import pandas as pd

from jlc.core.population_helpers import (
    redshift_from_lambda,
    luminosity_from_flux,
    rate_density_local,
)
from jlc.cosmology.lookup import AstropyCosmology
from jlc.engine.flux_grid import FluxGrid
from jlc.selection.base import SelectionModel
from jlc.population.schechter import SchechterLF
from jlc.labels.lae import LAELabel
from jlc.labels.oii import OIILabel
from jlc.types import SharedContext
from jlc.measurements.flux import FluxMeasurement
from jlc.labels.registry import LabelRegistry


def test_redshift_from_lambda_basic():
    # LAE 1215.67 -> z = lam/rest - 1
    rest = 1215.67
    lam = rest * (1.0 + 3.0)
    z = redshift_from_lambda(lam, rest)
    assert np.isclose(z, 3.0)
    # invalid inputs
    assert not np.isfinite(redshift_from_lambda(-1.0, rest))
    assert not np.isfinite(redshift_from_lambda(lam, -1.0))


ess = AstropyCosmology()

def test_luminosity_from_flux_matches_definition():
    z = 2.5
    F = np.array([1e-18, 2e-18, 5e-17])
    L = luminosity_from_flux(F, z, ess)
    # Expected L = 4*pi*dL^2*F with dL in cm
    dL_mpc = ess.luminosity_distance(z)
    MPC_TO_CM = 3.0856775814913673e24
    dL_cm = dL_mpc * MPC_TO_CM
    L_exp = 4.0 * np.pi * (dL_cm ** 2) * F
    assert np.allclose(L, L_exp)


def _make_ctx_and_registry():
    cosmo = AstropyCosmology()
    selection = SelectionModel(F50=2e-17, w=5e-18)
    fg = FluxGrid(Fmin=1e-19, Fmax=1e-14, n=64)
    caches = {"flux_grid": fg}
    config = {"use_rate_priors": True, "use_global_priors": False}
    ctx = SharedContext(cosmo=cosmo, selection=selection, caches=caches, config=config)
    lae_lf = SchechterLF(log10_Lstar=42.72, alpha=-1.75, log10_phistar=-3.2, Lmin=1e39, Lmax=1e46)
    oii_lf = SchechterLF(log10_Lstar=41.4, alpha=-1.2, log10_phistar=-2.4, Lmin=1e38, Lmax=1e45)
    flux_meas = FluxMeasurement()
    lae = LAELabel(lae_lf, selection, [flux_meas])
    oii = OIILabel(oii_lf, selection, [flux_meas])
    reg = LabelRegistry([lae, oii])
    return ctx, reg


def test_rate_density_local_matches_labels():
    ctx, reg = _make_ctx_and_registry()
    # Build some dummy rows across wavelength range for both labels
    rows = []
    # Use two wavelengths within LAE and OII observable bands
    rows.append(pd.Series({"wave_obs": 4000.0, "flux_hat": 1e-17, "flux_err": 5e-18}))
    rows.append(pd.Series({"wave_obs": 5000.0, "flux_hat": 1e-17, "flux_err": 5e-18}))

    # LAE comparison
    lae = reg.model("lae")
    for row in rows:
        r_lbl = lae.rate_density(row, ctx)
        r_h = rate_density_local(row, ctx, lae.rest_wave, lae.lf, ctx.selection)
        assert np.isfinite(r_lbl) and np.isfinite(r_h)
        assert np.isclose(r_lbl, r_h, rtol=1e-12, atol=0.0)

    # OII comparison
    oii = reg.model("oii")
    for row in rows:
        r_lbl = oii.rate_density(row, ctx)
        r_h = rate_density_local(row, ctx, oii.rest_wave, oii.lf, ctx.selection)
        assert np.isfinite(r_lbl) and np.isfinite(r_h)
        assert np.isclose(r_lbl, r_h, rtol=1e-12, atol=0.0)
