import numpy as np
import pandas as pd
from .base import MeasurementModule


class FluxMeasurement(MeasurementModule):
    name = "flux"
    catalog_columns = ("flux_hat", "flux_err")
    latent_key = "F_true"

    def log_likelihood(self, row: pd.Series, latent: dict, ctx) -> float:
        # latent must contain "F_true"
        F = float(latent.get("F_true", 0.0)) if latent is not None else 0.0
        mu = F
        # Allow an extra scatter term from noise_hyperparams (added in quadrature)
        sigma_obs = float(row.get("flux_err", 1.0))
        extra = 0.0
        try:
            extra = float(self.noise_hyperparams.get("extra_scatter", 0.0))
        except Exception:
            extra = 0.0
        sigma = float(np.hypot(sigma_obs, extra))
        x = float(row.get("flux_hat", 0.0))
        if sigma <= 0 or not np.isfinite(sigma):
            return -np.inf
        return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

    def simulate_observed(self, latent_value: float, ctx) -> dict:
        """Draw (flux_hat, flux_err) given latent true flux and context.

        Uses a fixed flux_err from ctx.config.get("flux_err_sim", ...) if available;
        otherwise leaves it to the caller to fill in.
        """
        try:
            err = float(getattr(ctx, "config", {}).get("flux_err_sim", 0.0))
        except Exception:
            err = 0.0
        err = max(err, 0.0)
        x = float(max(latent_value + (np.random.normal(0.0, err) if err > 0 else 0.0), 0.0))
        return {"flux_hat": x, "flux_err": err}
