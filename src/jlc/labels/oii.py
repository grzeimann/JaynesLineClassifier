import numpy as np
import pandas as pd
from dataclasses import dataclass
from .base import LabelModel
from jlc.population.schechter import SchechterLF


@dataclass
class OIIHyperparams:
    log10_Lstar: float = 41.4
    alpha: float = -1.2
    log10_phistar: float = -2.4
    Lmin: float | None = None
    Lmax: float | None = None


class OIILabel(LabelModel):
    """[O II] label model: rate prior + flux-marginalized evidence, self-contained.

    - rate_density(row, ctx): integrates Schechter LF × dV/dz × selection over flux at the row’s wavelength; returns observed-space rate per sr per Å.
    - extra_log_likelihood(row, ctx): measurement-only evidence marginalized over the shared FluxGrid with a neutral prior on F.
    - simulate_catalog(...): engine-aligned per-label simulator reusing the same ingredients for PPP generation and diagnostics.
    """
    label = "oii"
    rest_wave = 3727.0  # Angstrom
    hyperparam_cls = OIIHyperparams
    feature_names = ("wave_obs", "flux_hat", "flux_err")

    def __init__(self, lf: SchechterLF | None = None, selection_model=None, measurement_modules=None, *,
                 hyperparams: OIIHyperparams | dict | None = None, cosmology=None, noise_model=None, flux_grid=None):
        if hyperparams is None and lf is not None:
            hyperparams = OIIHyperparams(
                log10_Lstar=lf.log10_Lstar,
                alpha=lf.alpha,
                log10_phistar=lf.log10_phistar,
                Lmin=getattr(lf, "Lmin", None),
                Lmax=getattr(lf, "Lmax", None),
            )
        self.hyperparams = self._coerce_hyperparams(hyperparams)
        if lf is None:
            self.lf = SchechterLF(
                self.hyperparams.log10_Lstar,
                self.hyperparams.alpha,
                self.hyperparams.log10_phistar,
                Lmin=self.hyperparams.Lmin,
                Lmax=self.hyperparams.Lmax,
            )
        else:
            self.lf = lf
        self.cosmology = cosmology
        self.selection_model = selection_model
        self.selection = selection_model
        self.noise_model = noise_model
        self.flux_grid = flux_grid
        self.measurement_modules = list(measurement_modules or [])

    def _coerce_hyperparams(self, hp):
        if hp is None:
            return self.hyperparam_cls()
        if isinstance(hp, self.hyperparam_cls):
            return hp
        if isinstance(hp, dict):
            return self.hyperparam_cls(**hp)
        return self.hyperparam_cls()

    def rate_density(self, row: pd.Series, ctx) -> float:
        # Use shared helper for observed-space rate density integration (per sr per Å)
        from jlc.core.population_helpers import rate_density_local
        r = float(rate_density_local(row, ctx, self.rest_wave, self.lf, self.selection))
        # Optional factorized selection multiplier (neutral by default)
        try:
            use_fac = bool(getattr(ctx, "config", {}).get("use_factorized_selection", False))
        except Exception:
            use_fac = False
        if use_fac and self.selection is not None:
            try:
                wave_obs = float(row.get("wave_obs", np.nan))
                F_hat = float(row.get("flux_hat", np.nan))
                latent = {"F_true": float(F_hat) if np.isfinite(F_hat) else 0.0,
                          "wave_true": float(wave_obs) if np.isfinite(wave_obs) else float(self.rest_wave)}
                c_fac = float(self.selection.completeness_factorized(self, latent, self.measurement_modules, ctx))
                if np.isfinite(c_fac) and c_fac >= 0:
                    r *= c_fac
            except Exception:
                pass
        return r

    def extra_log_likelihood(self, row: pd.Series, ctx) -> float:
        """Measurement-only evidence marginalized over F with neutral prior."""
        z = float(row.get("wave_obs", np.nan)) / self.rest_wave - 1.0
        if not np.isfinite(z) or z <= 0:
            return -np.inf
        F_grid, log_w = ctx.caches["flux_grid"].grid(row)
        log_like = np.zeros_like(F_grid)
        for k, F in enumerate(F_grid):
            wave_true = self.rest_wave * (1.0 + float(z))
            latent = {"F_true": float(F), "z": float(z), "wave_true": float(wave_true)}
            ll = 0.0
            for m in self.measurement_modules:
                ll += float(m.log_likelihood(row, latent, ctx))
            log_like[k] = ll
        logZ = ctx.caches["flux_grid"].logsumexp(log_like + log_w)
        return float(logZ)

    # --- New per-label simulator (engine-aligned) ---
    def simulate_catalog(
        self,
        ctx,
        ra_low: float,
        ra_high: float,
        dec_low: float,
        dec_high: float,
        wave_min: float,
        wave_max: float,
        flux_err: float = 1e-17,
        f_lim: float | None = None,
        fake_rate_per_sr_per_A: float = 0.0,
        rng=None,
        nz: int = 256,
        snr_min: float | None = None,
    ):
        import numpy as _np
        import pandas as _pd
        from jlc.simulate.model_ppp import (
            PPPConfig,
            skybox_solid_angle_sr,
            _compute_label_grids,
            _rate_grid_per_sr,
            integrate_rate_over_flux,
            expected_count_from_rate,
            expected_volume_from_dVdz,
            sample_z_indices,
            sample_F_given_z,
        )
        try:
            if str(getattr(ctx, "config", {}).get("volume_mode", "real")).lower() == "virtual":
                return _pd.DataFrame(columns=["ra","dec","true_class","wave_obs","flux_true","flux_hat","flux_err"]).copy(), {
                    "label": self.label, "mu": 0.0, "V_Mpc3": 0.0
                }
        except Exception:
            pass
        rng = rng or _np.random.default_rng()
        omega = skybox_solid_angle_sr(ra_low, ra_high, dec_low, dec_high)
        cfg = PPPConfig(nz=int(nz), flux_grid=getattr(ctx, "caches", {}).get("flux_grid"))
        z_grid, F_grid, zmin, zmax = _compute_label_grids(ctx, self.rest_wave, wave_min, wave_max, cfg)
        if z_grid is None:
            return _pd.DataFrame(columns=["ra","dec","true_class","wave_obs","flux_true","flux_hat","flux_err"]).copy(), {
                "label": self.label, "mu": 0.0, "V_Mpc3": 0.0
            }
        rate, dVdz, dL2 = _rate_grid_per_sr(ctx, self.lf, ctx.selection, self.rest_wave, z_grid, F_grid)
        r_z = integrate_rate_over_flux(rate, F_grid)
        mu = expected_count_from_rate(r_z, z_grid, omega)
        V = expected_volume_from_dVdz(dVdz, z_grid, omega)
        # Diagnostics
        try:
            class _UnitSel:
                def completeness(self, F, wave):
                    return _np.ones_like(F, dtype=float)
            rate0, _dVdz0, _dL20 = _rate_grid_per_sr(ctx, self.lf, _UnitSel(), self.rest_wave, z_grid, F_grid)
            r_z0 = integrate_rate_over_flux(rate0, F_grid)
            mu0 = expected_count_from_rate(r_z0, z_grid, omega)
            S_eff = (mu / mu0) if (_np.isfinite(mu0) and mu0 > 0) else _np.nan
        except Exception:
            mu0 = float('nan'); S_eff = float('nan')
        n = int(max(0, _np.random.poisson(mu) if _np.isfinite(mu) and mu > 0 else 0))
        from jlc.simulate.model_ppp import _cap_events
        n = _cap_events(self.label, n, mu)
        if n <= 0:
            return _pd.DataFrame(columns=["ra","dec","true_class","wave_obs","flux_true","flux_hat","flux_err"]).copy(), {
                "label": self.label, "mu": float(mu), "V_Mpc3": float(V), "mu_no_selection": float(mu0), "S_eff": float(S_eff),
                "rest_wave": float(self.rest_wave), "zmin": float(zmin), "zmax": float(zmax)
            }
        idx_z, p_z = sample_z_indices(rng, z_grid, r_z, n)
        z_samp = z_grid[idx_z]
        lam_obs = self.rest_wave * (1.0 + z_samp)
        F_samp = _np.empty(n, dtype=float)
        for k in range(n):
            i = idx_z[k]
            F_samp[k] = sample_F_given_z(rng, rate[i, :], F_grid)
        F_err = _np.full(n, float(flux_err))
        F_hat = _np.clip(F_samp + rng.normal(0.0, F_err), 0.0, None)
        ra = rng.uniform(ra_low, ra_high, size=n)
        s1 = _np.sin(_np.deg2rad(dec_low)); s2 = _np.sin(_np.deg2rad(dec_high))
        u = rng.uniform(min(s1, s2), max(s1, s2), size=n)
        dec = _np.rad2deg(_np.arcsin(u))
        df = _pd.DataFrame({
            "ra": ra,
            "dec": dec,
            "true_class": _np.array([self.label]*n),
            "wave_obs": lam_obs,
            "flux_true": F_samp,
            "flux_hat": F_hat,
            "flux_err": F_err,
        })
        return df, {"label": self.label, "mu": float(mu), "V_Mpc3": float(V), "mu_no_selection": float(mu0), "S_eff": float(S_eff),
                    "rest_wave": float(self.rest_wave), "zmin": float(zmin), "zmax": float(zmax)}
