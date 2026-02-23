from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class MeasurementModule(ABC):
    """Base class for all measurement models.

    Encapsulates noise models and per-measurement priors.
    Provides likelihood, optional marginalization helpers, and simulation hooks.
    """
    # Short name used in configs and prior records
    name: str = "measurement"
    # Catalog columns this measurement reads/writes (advisory metadata)
    catalog_columns: Tuple[str, ...] = ()
    # Name of the latent true variable key in the 'latent' dict
    # e.g. 'F_true', 'wave_true', 'width_true'
    latent_key: str = ""

    def __init__(
        self,
        noise_hyperparams: Optional[Dict[str, Any]] = None,
        prior_hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize with noise + prior hyperparameters.

        noise_hyperparams: Parameters controlling the noise model (e.g. extra scatter).
        prior_hyperparams: Parameters controlling the prior over latent true values
                           for this measurement, conditional on a label.
        """
        self.noise_hyperparams = dict(noise_hyperparams or {})
        self.prior_hyperparams = dict(prior_hyperparams or {})

    # ---- Likelihood ----
    @abstractmethod
    def log_likelihood(self, row: pd.Series, latent: Dict[str, Any], ctx) -> float:
        """Return log p(measurement | latent_true, noise_hyperparams, ctx)."""
        raise NotImplementedError

    # ---- Prior over latent true for this measurement ----
    def sample_latent_prior(self, size: int, label, ctx):
        """Draw samples of latent[latent_key] from the prior for this label.

        Default: not implemented (measurements without extra latent can ignore).
        """
        raise NotImplementedError

    # Optional: analytic or MC marginalization helper
    def marginal_log_evidence(self, row: pd.Series, label, ctx) -> float:
        """Optional log p(measurement | label) with latent integrated out.

        Default: not implemented; labels can directly call log_likelihood with
        explicit latent values instead.
        """
        raise NotImplementedError

    # ---- Simulation ----
    def simulate_observed(self, latent_value: float, ctx) -> Dict[str, float]:
        """Draw observed catalog fields given latent true value.

        Returns a dict mapping catalog column names to values.
        Default: not implemented.
        """
        raise NotImplementedError
