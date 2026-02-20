import numpy as np
import pandas as pd
from .base import LabelModel
from jlc.types import EvidenceResult
from jlc.utils.constants import EPS_LOG


class FakeLabel(LabelModel):
    """Fake label model: contextual rate prior + flux-marginalized evidence.

    - rate_density(row, ctx): base fake rate per sr per Å, optionally a simple
      mixture over components (e.g., sky_residual, noise) and modulated by an
      empirical λ-PDF if available; multiplied by effective_search_measure.
    - extra_log_likelihood(row, ctx): measurement-only evidence marginalized
      over latent flux using the shared FluxGrid with a neutral prior on F.
    - simulate_catalog(...): per-label simulator consistent with the same
      ingredients used in rate_density; yields diagnostics like μ and band size.
    """
    label = "fake"

    def __init__(self, selection_model=None, measurement_modules=None, mu_ln_offset: float = -2.0, sigma_ln: float = 1.0,
                 use_mixture: bool = True):
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

        Phase 2 extension:
        - Optional contextual mixture for fake rate prior. If use_mixture=True (default),
          rate_density() computes a mixture over simple components and exposes per-component
          rates via rate_components(). Backward compatible: if disabled or features absent,
          behavior reduces to a uniform rate modulated by empirical λ-PDF when available.
        """
        self.selection = selection_model
        self.measurement_modules = list(measurement_modules or [])
        self.mu_ln_offset = float(mu_ln_offset)
        self.sigma_ln = float(sigma_ln)
        self.use_mixture = bool(use_mixture)

    # --- Mixture helpers (Phase 2) ---
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

    def _mixture_weights(self, row: pd.Series, ctx) -> dict[str, float]:
        """
        Return mixture weights π_k(x) that sum to 1.0. Uses contextual features if present.
        Defaults to uniform weights over two components: sky_residual and noise.
        """
        # Placeholder contextual hooks (future): sky_line_prox, dist_to_mask, ifu_flags, n_exp_support
        # For now, use uniform weights.
        return {"sky_residual": 0.5, "noise": 0.5}

    def rate_components(self, row: pd.Series, ctx) -> dict[str, float]:
        """Per-component fake rates (per sr per Å) for diagnostics. Safe and finite."""
        lam = float(row.get("wave_obs", np.nan))
        rho = self._base_rho(ctx)
        # component shapes s_k(row)
        s_sky = self._lambda_shape(lam, ctx)  # modulated by empirical λ-PDF
        s_noise = 1.0  # constant baseline
        shapes = {"sky_residual": s_sky, "noise": s_noise}
        pis = self._mixture_weights(row, ctx)
        # Effective search measure multiplier
        eff = 1.0
        try:
            from jlc.rates.observed_space import effective_search_measure
            eff = float(effective_search_measure(row, ctx))
        except Exception:
            pass
        # Combine into per-component intensities: r_k = rho * π_k * s_k * eff
        out = {}
        for k, pi in pis.items():
            sk = float(shapes.get(k, 1.0))
            val = max(rho * max(float(pi), 0.0) * max(sk, 0.0) * max(eff, 0.0), 0.0)
            out[k] = float(val if np.isfinite(val) else 0.0)
        return out

    def rate_density(self, row: pd.Series, ctx) -> float:
        """Contextual mixture fake rate per sr per Å.

        - Base rate from ctx.config["fake_rate_per_sr_per_A"].
        - Optionally modulated by empirical λ-PDF shape via a sky_residual component.
        - Multiplies by effective_search_measure.
        - Backward compatible: if use_mixture=False, reduces to legacy single-component behavior.
        """
        if not self.use_mixture:
            # Legacy single-component
            lam = float(row.get("wave_obs", np.nan))
            r = self._base_rho(ctx)
            r *= self._lambda_shape(lam, ctx)
            try:
                from jlc.rates.observed_space import effective_search_measure
                r *= float(effective_search_measure(row, ctx))
            except Exception:
                pass
            return r
        # Mixture path
        comps = self.rate_components(row, ctx)
        return float(sum(comps.values()))

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
            latent = {"F_true": float(F)}
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
