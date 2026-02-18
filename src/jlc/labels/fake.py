import numpy as np
import pandas as pd
from .base import LabelModel
from jlc.types import EvidenceResult


class FakeLabel(LabelModel):
    label = "fake"

    def __init__(self, selection_model=None, measurement_modules=None, mu_ln_offset: float = -2.0, sigma_ln: float = 1.0):
        """
        A generative Fake label consistent with the simulator assumptions.

        - Prior over wavelength is uniform on [wave_min, wave_max] (if provided in ctx.config),
          contributing a factor 1/Δλ. If unavailable, this factor is omitted.
        - Prior over true flux F_true is LogNormal with parameters derived from f_lim in ctx.config:
            ln F ~ Normal(mu=ln(max(f_lim,1e-20)) + mu_ln_offset, sigma=sigma_ln)
          Defaults match the PPP simulator (mu = ln(f_lim) - 2, sigma = 1).
        - Selection S(F, λ) is applied using the provided selection_model; if None, S=1.
        - Measurement likelihoods are applied via measurement_modules (e.g., FluxMeasurement).
        - Marginalization over F_true is performed on the shared FluxGrid cache.
        """
        self.selection = selection_model
        self.measurement_modules = list(measurement_modules or [])
        self.mu_ln_offset = float(mu_ln_offset)
        self.sigma_ln = float(sigma_ln)

    def _flux_logprior(self, F: np.ndarray, f_lim: float | None) -> np.ndarray:
        # LogNormal prior density over F (in linear flux units)
        base = max(float(f_lim) if f_lim is not None else 1e-17, 1e-20)
        mu = np.log(base) + self.mu_ln_offset
        sigma = self.sigma_ln
        # log pdf of lognormal
        F = np.asarray(F, dtype=float)
        F_safe = np.clip(F, 1e-300, None)
        return -0.5 * ((np.log(F_safe) - mu) / sigma) ** 2 - np.log(F_safe * sigma * np.sqrt(2 * np.pi))

    def log_evidence(self, row: pd.Series, ctx) -> EvidenceResult:
        lam = float(row.get("wave_obs", np.nan))
        # Wavelength prior: uniform in [wave_min, wave_max]
        wave_min = ctx.config.get("wave_min") if hasattr(ctx, "config") else None
        wave_max = ctx.config.get("wave_max") if hasattr(ctx, "config") else None
        if wave_min is not None and wave_max is not None and wave_max > wave_min:
            log_p_lambda = -np.log(float(wave_max - wave_min))
        else:
            log_p_lambda = 0.0  # unknown band; treat as constant w.r.t. labels

        # Flux grid and integration weights
        F_grid, log_w = ctx.caches["flux_grid"].grid(row)

        # Prior over F
        f_lim = ctx.config.get("f_lim") if hasattr(ctx, "config") else None
        log_p_F = self._flux_logprior(F_grid, f_lim)

        # Selection S(F, λ)
        if self.selection is not None and np.isfinite(lam):
            S = self.selection.completeness(F_grid, lam)
        else:
            S = np.ones_like(F_grid)
        log_S = np.log(S + 1e-300)

        log_prior = log_p_lambda + log_p_F + log_S

        # Measurement likelihoods
        log_like = np.zeros_like(F_grid)
        for k, F in enumerate(F_grid):
            latent = {"F_true": float(F)}
            ll = 0.0
            for m in self.measurement_modules:
                ll += float(m.log_likelihood(row, latent, ctx))
            log_like[k] = ll

        logZ = ctx.caches["flux_grid"].logsumexp(log_prior + log_like + log_w)
        return EvidenceResult(self.label, float(logZ), {})
