import numpy as np
import pandas as pd
from .base import LabelModel
from jlc.types import EvidenceResult


class SimpleFakeDensity:
    """A very simple empirical-like density placeholder.

    Returns a finite logpdf based on a broad normal over flux_hat with sigma from flux_err.
    If flux_err is missing, uses a large default to remain non-informative.
    """

    def logpdf(self, row: pd.Series) -> float:
        x = float(row.get("flux_hat", 0.0))
        sigma = float(row.get("flux_err", 5.0))
        sigma = max(sigma, 1e-6)
        return -0.5 * (x / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))


class FakeLabel(LabelModel):
    label = "fake"

    def __init__(self, fake_density_model: SimpleFakeDensity | None = None):
        self.fake_density_model = fake_density_model or SimpleFakeDensity()
        self.measurement_modules = []

    def log_evidence(self, row: pd.Series, ctx) -> EvidenceResult:
        logp = float(self.fake_density_model.logpdf(row))
        return EvidenceResult(self.label, logp, {})
