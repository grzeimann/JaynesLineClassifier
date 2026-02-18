import numpy as np
import pandas as pd
from .base import LabelModel
from jlc.types import EvidenceResult
from jlc.population.schechter import SchechterLF


class OIILabel(LabelModel):
    label = "oii"
    rest_wave = 3727.0  # Angstrom

    def __init__(self, lf: SchechterLF, selection_model, measurement_modules):
        self.lf = lf
        self.selection = selection_model
        self.measurement_modules = list(measurement_modules)

    def log_evidence(self, row: pd.Series, ctx) -> EvidenceResult:
        # 1) map wavelength -> redshift
        z = float(row.get("wave_obs", np.nan)) / self.rest_wave - 1.0
        if not np.isfinite(z) or z <= 0:
            return EvidenceResult(self.label, -np.inf, {"reason": "z<=0 or invalid"})

        # 2) marginalize over F_true on a 1D grid (Phase 1 strategy)
        F_grid, log_w = ctx.caches["flux_grid"].grid(row)

        # 3) prior density for each grid point (LF × volume × selection)
        dL = ctx.cosmo.luminosity_distance(z)
        L_grid = 4 * np.pi * (dL ** 2) * F_grid

        log_phi = np.log(self.lf.phi(L_grid) + 1e-300)
        log_dV = np.log(ctx.cosmo.dV_dz(z) + 1e-300)
        log_sel = np.log(self.selection.completeness(F_grid, float(row.get("wave_obs", np.nan))) + 1e-300)

        log_jac = -np.log(self.rest_wave)  # |dz/dlambda|

        log_prior = log_phi + log_dV + log_sel + log_jac

        # 4) measurement likelihoods for each grid point
        log_like = np.zeros_like(F_grid)
        for k, F in enumerate(F_grid):
            latent = {"F_true": float(F), "z": float(z)}
            ll = 0.0
            for m in self.measurement_modules:
                ll += float(m.log_likelihood(row, latent, ctx))
            log_like[k] = ll

        # 5) log-evidence via log-sum-exp including integration weights
        logZ = ctx.caches["flux_grid"].logsumexp(log_prior + log_like + log_w)
        return EvidenceResult(self.label, float(logZ), {"z": float(z)})
