import numpy as np
import pandas as pd
import math

from jlc.measurements.wavelength import WavelengthMeasurement


def test_wavelength_likelihood_neutral_when_no_error():
    m = WavelengthMeasurement()
    row = pd.Series({"wave_obs": 5000.0, "wave_err": 0.0})
    latent = {"wave_true": 4999.5}
    # With zero error, the module should be neutral (0.0 log-like contribution)
    ll = m.log_likelihood(row, latent, ctx={})
    assert ll == 0.0


def test_wavelength_likelihood_gaussian_value():
    m = WavelengthMeasurement()
    w_true = 5000.0
    w_obs = 5000.5
    sigma = 1.2
    row = pd.Series({"wave_obs": w_obs, "wave_err": sigma})
    latent = {"wave_true": w_true}
    ll = m.log_likelihood(row, latent, ctx={})
    # Expected Gaussian log pdf
    resid = (w_obs - w_true) / sigma
    expected = -0.5 * resid * resid - math.log(sigma * math.sqrt(2.0 * math.pi))
    assert np.isfinite(ll)
    assert abs(ll - expected) < 1e-10
