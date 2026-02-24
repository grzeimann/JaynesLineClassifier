import numpy as np
import pandas as pd
from dataclasses import dataclass
from .base import LabelModel
from jlc.population.schechter import SchechterLF
from jlc.utils.logging import log
from jlc.core.population_helpers import rate_density_local


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
        """Observed-space prior rate density r(λ) for this label at the row’s wavelength.

        Definition and units:
        - Returns the per-wavelength rate density r(λ) with units counts per (sr · Å).
        - r(λ) is obtained by integrating, over latent flux F, the per-flux integrand
              r_F(F, λ) = dV/dz · [ϕ(L(F,z)) · dL/dF] · S(F, λ) · |dz/dλ|
          which has units counts per (sr · Å · flux).
        - The Jacobian |dz/dλ| = 1/rest_wave is applied inside the central helpers
          that this method delegates to for consistency across labels.

        Implementation notes:
        - Uses population_helpers.rate_density_local with this label’s Schechter LF
          (self.lf), selection model (self.selection), and the cosmology from ctx.
        - The observed wavelength is taken from row['wave_obs']; if invalid, the
          helper returns 0.0.

        Parameters
        ----------
        row : pandas.Series
            Must contain at least 'wave_obs' (Å). Other fields may be used by
            the selection model or measurement modules.
        ctx : object
            Execution context with caches (e.g., FluxGrid) and cosmology.

        Returns
        -------
        float
            The rate density r(λ) in counts per (sr · Å) at the row’s wavelength.
        """
        return float(rate_density_local(row, ctx, self.rest_wave, self.lf, self.selection, label_name=self.label))

    def extra_log_likelihood(self, row: pd.Series, ctx) -> float:
        """Flux-marginalized measurement evidence log p(data | label, ctx).

        What this computes:
        - Returns the log-evidence after integrating out latent true flux F on
          the shared FluxGrid with a neutral prior (encoded in the grid’s
          quadrature weights). No population/LF prior is applied here.
        - For each grid value F, this sums measurement module log-likelihoods
          m.log_likelihood(row, latent, ctx), with latent containing F_true=F
          and the deterministic wavelength from z.
        - Integration is done in log-space via FluxGrid.logsumexp using the
          grid’s log-weights for numerical stability.

        Inputs/assumptions:
        - row must contain 'wave_obs' (Å). This sets z via z = wave_obs / rest_wave - 1.
          If z is invalid (non-finite or <= 0), returns -inf to exclude impossible cases.
        - ctx.caches['flux_grid'] must provide (F_grid, log_w) via .grid(row), and a
          stable .logsumexp utility.

        Output and units:
        - Returns a scalar float equal to log ∫ dF p(data | F, z, ctx) w(F), where w(F)
          are neutral quadrature weights. This is a dimensionless log-probability
          contribution independent of the population-rate prior (rate_density).
        """
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
            expected_count_from_rate,
            expected_volume_from_dVdz,
            sample_z_indices,
            sample_F_given_z,
        )
        from jlc.core.population_helpers import (
            rate_density_integrand_per_flux,
            integrate_over_flux,
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
        # Compute rates using population_helpers in observed space (per sr per Å)
        lam_grid = self.rest_wave * (1.0 + z_grid)
        Nz = z_grid.size
        Nf = F_grid.size
        r_lambda = _np.zeros(Nz, dtype=float)  # per sr per Å
        dVdz = _np.zeros(Nz, dtype=float)
        rate = _np.zeros((Nz, Nf), dtype=float)  # store r_F(F,λ) for sampling
        for i, z in enumerate(z_grid):
            lam_i = float(lam_grid[i])
            rF_i, dVdz_i, _dL2_i = rate_density_integrand_per_flux(
                self.lf, ctx.selection, F_grid, lam_i, self.rest_wave, float(z), ctx.cosmo, label_name=self.label
            )
            rate[i, :] = _np.asarray(rF_i, dtype=float)
            dVdz[i] = float(dVdz_i)
            r_lambda[i] = float(integrate_over_flux(rF_i, F_grid))
        # Expected counts and volume
        mu_per_sr = float(_np.trapz(r_lambda, x=lam_grid))
        mu = float(mu_per_sr * float(omega))
        V = expected_volume_from_dVdz(dVdz, z_grid, omega)
        # Build r(z) per sr per dz for z-sampling
        r_z = r_lambda * float(self.rest_wave)
        # --- Diagnostics: 50"×50" box summary
        dlam = float(lam_grid[-1] - lam_grid[0]) if lam_grid.size >= 2 else float("nan")
        rbar = (mu_per_sr / dlam) if (_np.isfinite(dlam) and dlam > 0) else float("nan")
        arcsec_to_rad = _np.deg2rad(1.0/3600.0)
        side_rad = 50.0 * arcsec_to_rad
        omega_box = float(side_rad * side_rad)
        mu_box = mu_per_sr * omega_box
        rbar_box = rbar * omega_box
        log(
            f"[oii.simulate_catalog] 50\"×50\" box (Ω_box≈{omega_box:.3e} sr): counts={mu_box:.3e}; rates per Å in box: r̄={rbar_box:.3e}"
        )
        from jlc.simulate.model_ppp import _poisson_safe, _cap_events
        n_draw = _poisson_safe(rng, mu) if (_np.isfinite(mu) and mu > 0) else 0
        n = _cap_events(self.label, int(max(0, n_draw)), mu)
        if n <= 0:
            return _pd.DataFrame(columns=["ra","dec","true_class","wave_obs","flux_true","flux_hat","flux_err"]).copy(), {
                "label": self.label, "mu": float(mu), "V_Mpc3": float(V),
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
        return df, {"label": self.label, "mu": float(mu), "V_Mpc3": float(V),
                    "rest_wave": float(self.rest_wave), "zmin": float(zmin), "zmax": float(zmax)}
