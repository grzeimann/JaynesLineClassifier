import numpy as np
import pandas as pd
from dataclasses import dataclass
from .base import LabelModel
from jlc.types import EvidenceResult
from jlc.utils.constants import EPS_LOG


@dataclass
class FakeHyperparams:
    # Use linear rate density to align with current config; allow None to defer to ctx.config
    rate_per_sr_per_A: float | None = None
    mu_ln_offset: float = -2.0
    sigma_ln: float = 1.0


class FakeLabel(LabelModel):
    """Fake label model: homogeneous λ-rate with flux-conditioned prior.

    - rate_density(row, ctx): r_fake(λ, F_hat) = ρ(λ) · ∫ dF p_fake(F) S(F,λ) p(F_hat|F)
      where ρ(λ) is a base per-(sr·Å) rate optionally modulated by an empirical λ-PDF,
      p_fake(F) is a LogNormal prior over true flux, S is selection completeness, and
      p(F_hat|F) is the Gaussian flux measurement model from flux_err. Multiplies by
      effective_search_measure(row, ctx).
    - extra_log_likelihood(row, ctx): measurement-only evidence marginalized over
      latent flux using the shared FluxGrid with a neutral prior on F.
    - simulate_catalog(...): per-label simulator consistent with the same ingredients
      used in rate_density; yields diagnostics like μ and band size.
    """
    label = "fake"
    rest_wave = None
    hyperparam_cls = FakeHyperparams
    feature_names = ("wave_obs", "flux_hat", "flux_err")

    def __init__(self, selection_model=None, measurement_modules=None, mu_ln_offset: float = -2.0, sigma_ln: float = 1.0, *,
                 hyperparams: FakeHyperparams | dict | None = None, cosmology=None, noise_model=None, flux_grid=None):
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
        # Build hyperparams: prefer explicit container, else from provided offsets
        if hyperparams is None:
            hyperparams = FakeHyperparams(rate_per_sr_per_A=None, mu_ln_offset=float(mu_ln_offset), sigma_ln=float(sigma_ln))
        self.hyperparams = self._coerce_hyperparams(hyperparams)
        # Back-compat attribute aliases used throughout methods
        try:
            self.mu_ln_offset = float(self.hyperparams.mu_ln_offset)
        except Exception:
            self.mu_ln_offset = -2.0
        try:
            self.sigma_ln = float(self.hyperparams.sigma_ln)
        except Exception:
            self.sigma_ln = 1.0
        # Standardized attachments
        self.cosmology = cosmology
        self.selection_model = selection_model
        self.selection = selection_model  # backward compat
        self.noise_model = noise_model
        self.flux_grid = flux_grid
        self.measurement_modules = list(measurement_modules or [])

    # Hyperparameter helpers
    def _coerce_hyperparams(self, hp):
        if hp is None:
            return self.hyperparam_cls()
        if isinstance(hp, self.hyperparam_cls):
            return hp
        if isinstance(hp, dict):
            return self.hyperparam_cls(**hp)
        return self.hyperparam_cls()

    # --- Rate helpers ---
    def _base_rho(self, ctx) -> float:
        try:
            cfg = getattr(ctx, "config", {}) or {}
            return max(float(cfg.get("fake_rate_per_sr_per_A", 0.0)), 0.0)
        except Exception:
            return 0.0

    def _lambda_shape(self, lam: float, ctx) -> float:
        try:
            cache = getattr(ctx, "caches", {}).get("fake_lambda_pdf")
            if cache is None:
                return 1.0
            from jlc.rates.observed_space import eval_fake_lambda_shape
            s = float(eval_fake_lambda_shape(lam, cache))
            if np.isfinite(s) and s > 0:
                return s
            return 1.0
        except Exception:
            return 1.0

    def rate_density(self, row: pd.Series, ctx) -> float:
        """Flux-conditioned fake rate per sr per Å.

        r_fake(λ, F_hat) = ρ(λ) · ∫ dF_true p_fake(F_true) · S(F_true,λ) · p(F_hat|F_true)
        where p_fake is the lognormal prior configured by (mu_ln_offset, sigma_ln),
        S is selection completeness, and p(F_hat|F_true) is Gaussian with sigma=flux_err.
        Multiplies by effective_search_measure(row, ctx).
        """
        # Base λ-shape
        try:
            lam = float(row.get("wave_obs", np.nan))
        except Exception:
            lam = float("nan")
        if not (np.isfinite(lam) and lam > 0):
            return 0.0
        rho = self._base_rho(ctx)
        if rho <= 0 or not np.isfinite(rho):
            return 0.0
        rho_lambda = rho * self._lambda_shape(lam, ctx)
        # Measurement
        try:
            F_hat = float(row.get("flux_hat", np.nan))
            sigma = float(row.get("flux_err", np.nan))
        except Exception:
            F_hat, sigma = float("nan"), float("nan")
        if not (np.isfinite(F_hat) and F_hat >= 0):
            return 0.0
        # Flux prior parameters
        try:
            f_lim = getattr(ctx, "config", {}).get("f_lim", None)
            base = max(float(f_lim) if f_lim is not None else 1e-17, 1e-20)
        except Exception:
            base = 1e-17
        mu_ln = np.log(base) + self.mu_ln_offset
        sigma_ln = self.sigma_ln
        # Build F_true grid from shared FluxGrid; fallback to local window around F_hat
        F_grid_true = None
        try:
            fg = getattr(getattr(ctx, "caches", {}), "get", lambda k, d=None: None)("flux_grid")
            if fg is not None and hasattr(fg, "grid"):
                Fg, _w = fg.grid(row)
                Fg = np.asarray(Fg, dtype=float)
                Fg = Fg[np.isfinite(Fg) & (Fg >= 0.0)]
                if Fg.size >= 2:
                    F_grid_true = Fg
        except Exception:
            F_grid_true = None
        if F_grid_true is None:
            # local window
            nsig = getattr(getattr(ctx, "config", {}), "get", lambda k, d=None: None)("flux_marg_nsigma")
            try:
                nsig = float(nsig if nsig is not None else 6.0)
            except Exception:
                nsig = 6.0
            nsig = 6.0 if (not np.isfinite(nsig) or nsig <= 0) else nsig
            F_min = max(0.0, F_hat - nsig * max(sigma, 0.0 if not np.isfinite(sigma) else sigma))
            F_max = F_hat + nsig * (sigma if np.isfinite(sigma) else 0.0)
            span = F_max - F_min
            if not (np.isfinite(span) and span > 0):
                F_min = max(0.0, F_hat)
                F_max = F_min + max(sigma if np.isfinite(sigma) else 1e-30, 1e-30)
            Nf = getattr(getattr(ctx, "config", {}), "get", lambda k, d=None: None)("flux_marg_npts")
            try:
                Nf = int(Nf if Nf is not None else 256)
            except Exception:
                Nf = 256
            Nf = max(int(Nf), 32)
            F_grid_true = np.linspace(F_min, F_max, Nf, dtype=float)
        # Selection completeness
        sel = self.selection
        try:
            S = sel.completeness(F_grid_true, float(lam)) if sel is not None else np.ones_like(F_grid_true)
            S = np.clip(np.asarray(S, dtype=float), 0.0, 1.0)
        except Exception:
            S = np.ones_like(F_grid_true)
        # Prior over F_true (lognormal pdf in linear F)
        F_safe = np.clip(F_grid_true, EPS_LOG, None)
        log_pF = -0.5 * ((np.log(F_safe) - mu_ln) / sigma_ln) ** 2 - np.log(F_safe * sigma_ln * np.sqrt(2 * np.pi))
        pF = np.exp(log_pF)
        # Measurement model p(F_hat | F_true)
        if np.isfinite(sigma) and sigma > 0:
            inv_s = 1.0 / sigma
            norm = inv_s / np.sqrt(2.0 * np.pi)
            resid = (F_hat - F_grid_true) * inv_s
            p_meas = norm * np.exp(-0.5 * resid * resid)
        else:
            # delta-like: evaluate at F_hat by nearest grid bin
            idx = int(np.argmin(np.abs(F_grid_true - F_hat)))
            p_meas = np.zeros_like(F_grid_true)
            p_meas[idx] = 1.0 / max(np.gradient(F_grid_true)[idx], 1e-30)  # approximate delta density
        # Integrate
        integrand = pF * S * p_meas
        factor = float(np.trapz(integrand, x=F_grid_true))
        if not np.isfinite(factor) or factor < 0:
            factor = 0.0
        r = rho_lambda * factor
        # Effective search measure
        try:
            from jlc.rates.observed_space import effective_search_measure
            r *= float(effective_search_measure(row, ctx))
        except Exception:
            pass
        # Optional factorized selection multiplier (neutral by default)
        try:
            use_fac = bool(getattr(ctx, "config", {}).get("use_factorized_selection", False))
        except Exception:
            use_fac = False
        if use_fac and self.selection is not None:
            try:
                wave_obs = float(row.get("wave_obs", np.nan))
                F_hat = float(row.get("flux_hat", np.nan))
                latent_fac = {"F_true": float(F_hat) if np.isfinite(F_hat) else 0.0,
                              "wave_true": float(wave_obs) if np.isfinite(wave_obs) else float(wave_obs)}
                c_fac = float(self.selection.completeness_factorized(self, latent_fac, self.measurement_modules, ctx))
                if np.isfinite(c_fac) and c_fac >= 0:
                    r *= c_fac
            except Exception:
                pass
        return float(max(r, 0.0))

    def _flux_logprior(self, F: np.ndarray, f_lim: float | None) -> np.ndarray:
        # LogNormal prior density over F (in linear flux units)
        base = max(float(f_lim) if f_lim is not None else 1e-17, 1e-20)
        mu = np.log(base) + self.mu_ln_offset
        sigma = self.sigma_ln
        # log pdf of lognormal
        F = np.asarray(F, dtype=float)
        F_safe = np.clip(F, EPS_LOG, None)
        return -0.5 * ((np.log(F_safe) - mu) / sigma) ** 2 - np.log(F_safe * sigma * np.sqrt(2 * np.pi))

    def extra_log_likelihood(self, row: pd.Series, ctx) -> float:
        """Measurement-only evidence marginalized over F with neutral prior (no z)."""
        F_grid, log_w = ctx.caches["flux_grid"].grid(row)
        log_like = np.zeros_like(F_grid)
        for k, F in enumerate(F_grid):
            # Provide deterministic wavelength latent so wavelength measurement can contribute
            try:
                wave_obs = float(row.get("wave_obs", np.nan))
            except Exception:
                wave_obs = float("nan")
            latent = {"F_true": float(F), "wave_true": float(wave_obs)}
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
        from jlc.simulate.model_ppp import skybox_solid_angle_sr, _cap_events
        rng = rng or _np.random.default_rng()
        # Base rate and expected count over sky x wavelength band
        rho = self._base_rho(ctx)
        omega = skybox_solid_angle_sr(ra_low, ra_high, dec_low, dec_high)
        band = max(0.0, float(wave_max - wave_min))
        mu = float(rho * omega * band)
        n = _cap_events(self.label, int(max(0, _np.random.poisson(mu) if _np.isfinite(mu) and mu > 0 else 0)), mu)
        if n <= 0:
            return _pd.DataFrame(columns=["ra","dec","true_class","wave_obs","flux_true","flux_hat","flux_err"]).copy(), {
                "label": self.label, "mu": float(mu), "band_A": float(band), "rho": float(rho)
            }
        # Sample sky uniformly per steradian
        ra = rng.uniform(ra_low, ra_high, size=n)
        s1 = _np.sin(_np.deg2rad(dec_low)); s2 = _np.sin(_np.deg2rad(dec_high))
        u = rng.uniform(min(s1, s2), max(s1, s2), size=n)
        dec = _np.rad2deg(_np.arcsin(u))
        # Sample wavelength uniformly over band
        lam = rng.uniform(wave_min, wave_max, size=n)
        # Flux prior: log-normal around f_lim baseline
        base = max(float(f_lim) if f_lim is not None else 1e-17, 1e-20)
        mu_ln = _np.log(base) + self.mu_ln_offset
        sigma_ln = self.sigma_ln
        F_true = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n)
        F_err = _np.full(n, float(flux_err))
        F_hat = _np.clip(F_true + rng.normal(0.0, F_err), 0.0, None)
        # Apply selection acceptance per event
        C = _np.empty(n, dtype=float)
        sel = self.selection
        for i in range(n):
            try:
                ci = sel.completeness(_np.asarray([F_true[i]], dtype=float), float(lam[i]))
                C[i] = float(ci[0]) if _np.size(ci) > 0 else 0.0
            except Exception:
                C[i] = 1.0
        C = _np.clip(C, 0.0, 1.0)
        keep = rng.uniform(0.0, 1.0, size=n) < C
        if not _np.any(keep):
            return _pd.DataFrame(columns=["ra","dec","true_class","wave_obs","flux_true","flux_hat","flux_err"]).copy(), {
                "label": self.label, "mu": float(mu), "band_A": float(band), "rho": float(rho)
            }
        ra = ra[keep]; dec = dec[keep]; lam = lam[keep]
        F_true = F_true[keep]; F_err = F_err[keep]; F_hat = F_hat[keep]
        df = _pd.DataFrame({
            "ra": ra,
            "dec": dec,
            "true_class": _np.array([self.label]*ra.size),
            "wave_obs": lam,
            "flux_true": F_true,
            "flux_hat": F_hat,
            "flux_err": F_err,
        })
        return df, {"label": self.label, "mu": float(mu), "band_A": float(band), "rho": float(rho)}
