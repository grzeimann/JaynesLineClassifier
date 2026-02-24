import numpy as np
import pandas as pd
from .base import MeasurementModule


class WavelengthMeasurement(MeasurementModule):
    name = "wavelength"
    catalog_columns = ("wave_obs", "wave_err")
    latent_key = "wave_true"

    def log_likelihood(self, row: pd.Series, latent: dict, ctx) -> float:
        """Gaussian likelihood for observed wavelength given latent true wavelength.

        If wave_err is missing or non-positive, return 0.0 (neutral contribution)
        to avoid penalizing rows lacking wavelength uncertainties.

        If the configured prior type for wavelength is 'deterministic-from-z',
        this module is neutral (returns 0.0) to avoid double-counting a
        delta-like constraint implied by the label hypothesis.
        """
        # If prior indicates deterministic-from-z, be neutral regardless of inputs
        try:
            ptype = str(getattr(self, "prior_hyperparams", {}).get("_type", "")).strip().lower()
            if ptype in ("deterministic-from-z", "deterministic_from_z", "deterministic"):
                return 0.0
        except Exception:
            pass
        try:
            w_true = float(latent.get(self.latent_key, np.nan)) if latent is not None else np.nan
        except Exception:
            w_true = np.nan
        try:
            w_obs = float(row.get("wave_obs", np.nan))
            sigma_obs = float(row.get("wave_err", np.nan))
        except Exception:
            w_obs, sigma_obs = np.nan, np.nan
        # If we don't have a usable sigma, treat as neutral (no information)
        if not (np.isfinite(sigma_obs) and sigma_obs > 0):
            return 0.0
        if not (np.isfinite(w_obs) and np.isfinite(w_true)):
            return -np.inf
        resid = (w_obs - w_true) / sigma_obs
        return -0.5 * resid * resid - np.log(sigma_obs * np.sqrt(2.0 * np.pi))

    def simulate_observed(self, latent_value: float, ctx) -> dict:
        """Draw wave_obs, wave_err given latent wave_true.

        Uses ctx.config["wave_err_sim"] if provided; otherwise sets zero error
        and returns wave_obs=wave_true (deterministic).
        """
        try:
            err = float(getattr(ctx, "config", {}).get("wave_err_sim", 0.0))
        except Exception:
            err = 0.0
        err = max(err, 0.0)
        if err > 0:
            obs = float(latent_value + np.random.normal(0.0, err))
        else:
            obs = float(latent_value)
        return {"wave_obs": obs, "wave_err": err}
