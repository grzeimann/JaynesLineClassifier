import numpy as np
import pandas as pd
from .base import LabelModel
from jlc.types import EvidenceResult
from jlc.population.schechter import SchechterLF

# Cosmology returns d_L in Mpc; convert to cm for L = 4π d_L^2 F in CGS.
MPC_TO_CM = 3.085677581491367e24


class OIILabel(LabelModel):
    label = "oii"
    rest_wave = 3727.0  # Angstrom

    def __init__(self, lf: SchechterLF, selection_model, measurement_modules):
        self.lf = lf
        self.selection = selection_model
        self.measurement_modules = list(measurement_modules)

    def rate_density(self, row: pd.Series, ctx) -> float:
        # Respect virtual-volume mode: no physical sources in virtual runs
        try:
            if str(getattr(ctx, "config", {}).get("volume_mode", "real")).lower() == "virtual":
                return 0.0
        except Exception:
            pass
        lam = float(row.get("wave_obs", np.nan))
        if not np.isfinite(lam) or lam <= 0:
            return 0.0
        z = lam / self.rest_wave - 1.0
        if not np.isfinite(z) or z <= 0:
            return 0.0
        F_grid, _ = ctx.caches["flux_grid"].grid(row)
        dL = ctx.cosmo.luminosity_distance(z)
        dL_cm = dL * MPC_TO_CM
        dVdz = ctx.cosmo.dV_dz(z)
        dLdF = 4 * np.pi * (dL_cm ** 2)
        L_grid = 4 * np.pi * (dL_cm ** 2) * F_grid
        phi = self.lf.phi(L_grid)
        S = self.selection.completeness(F_grid, lam)
        jac = 1.0 / self.rest_wave
        rF = dVdz * (phi * dLdF) * S * jac
        r = float(np.trapz(rF, x=F_grid))
        try:
            from jlc.rates.observed_space import effective_search_measure
            r *= float(effective_search_measure(row, ctx))
        except Exception:
            pass
        return max(r, 0.0)

    def log_evidence(self, row: pd.Series, ctx) -> EvidenceResult:
        # Evidence uses only measurement likelihood marginalized over F with neutral prior.
        z = float(row.get("wave_obs", np.nan)) / self.rest_wave - 1.0
        if not np.isfinite(z) or z <= 0:
            return EvidenceResult(self.label, -np.inf, {"reason": "z<=0 or invalid"})

        F_grid, log_w = ctx.caches["flux_grid"].grid(row)

        log_like = np.zeros_like(F_grid)
        for k, F in enumerate(F_grid):
            latent = {"F_true": float(F), "z": float(z)}
            ll = 0.0
            for m in self.measurement_modules:
                ll += float(m.log_likelihood(row, latent, ctx))
            log_like[k] = ll

        logZ = ctx.caches["flux_grid"].logsumexp(log_like + log_w)
        return EvidenceResult(self.label, float(logZ), {"z": float(z)})
