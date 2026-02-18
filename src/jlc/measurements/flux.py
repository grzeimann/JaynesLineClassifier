import numpy as np
import pandas as pd
from .base import MeasurementModule


class FluxMeasurement(MeasurementModule):
    name = "flux"

    def log_likelihood(self, row: pd.Series, latent: dict, ctx) -> float:
        # latent must contain "F_true"
        F = float(latent["F_true"]) if latent is not None else 0.0
        mu = F
        sigma = float(row.get("flux_err", 1.0))
        x = float(row.get("flux_hat", 0.0))
        if sigma <= 0 or not np.isfinite(sigma):
            return -np.inf
        return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
