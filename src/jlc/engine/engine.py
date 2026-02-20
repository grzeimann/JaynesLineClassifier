import numpy as np
import pandas as pd
from jlc.utils.constants import EPS_LOG


class JaynesianEngine:
    def __init__(self, registry, ctx):
        self.registry = registry
        self.ctx = ctx

    def compute_extra_log_likelihood_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-label measurement-only evidence (flux-marginalized).

        Preferred API: evaluate each label's extra_log_likelihood(row, ctx).
        Output columns retain the historical naming convention logZ_<label> to
        avoid breaking downstream consumers; documentation clarifies that these
        derive from extra_log_likelihood under the refactored architecture.
        """
        out = df.copy()
        for label in self.registry.labels:
            model = self.registry.model(label)
            vals = []
            for _, row in df.iterrows():
                try:
                    v = model.extra_log_likelihood(row, self.ctx)
                    val = float(v)
                    if not np.isfinite(val):
                        val = -np.inf
                except Exception:
                    val = -np.inf
                vals.append(val)
            out[f"logZ_{label}"] = vals
        return out

    def compute_log_evidence_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deprecated: use compute_extra_log_likelihood_matrix instead.

        This shim emits a DeprecationWarning and forwards to the new API.
        """
        import warnings
        warnings.warn(
            "JaynesianEngine.compute_log_evidence_matrix is deprecated; use compute_extra_log_likelihood_matrix()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_extra_log_likelihood_matrix(df)

    def compute_posteriors(self, df: pd.DataFrame, log_prior_weights: dict | None = None, mode: str | None = None) -> pd.DataFrame:
        """Compute per-label evidences and return a DataFrame with posteriors and diagnostics.

        Parameters
        ----------
        df : pandas.DataFrame
            Input catalog with at least wave_obs, flux_hat, flux_err; additional columns may be used by labels.
        log_prior_weights : dict | None
            Optional global per-label log-weights (e.g., from PPP expected counts). Keys must match registry.labels.
        mode : {"rate_only", "likelihood_only", "rate_times_likelihood"} or None
            Posterior combination mode (case-insensitive). If None, uses ctx.config.get("engine_mode", "rate_times_likelihood").

        Returns
        -------
        pandas.DataFrame
            Copy of df with added columns:
            - logZ_<label>: measurement-only evidence from extra_log_likelihood
            - rate_<label>: observed-space rate density for each label
            - p_<label>: posterior probability per label
            - Additional diagnostics (e.g., totals, prior flags), depending on configuration.
        """
        evid = self.compute_extra_log_likelihood_matrix(df)
        return self.normalize_posteriors(evid, log_prior_weights=log_prior_weights, mode=mode)

    def normalize_posteriors(self, df: pd.DataFrame, log_prior_weights: dict | None = None, mode: str | None = None) -> pd.DataFrame:
        out = df.copy()
        labels = self.registry.labels

        # Config toggles
        cfg = getattr(self.ctx, "config", {}) or {}
        use_rate_priors = bool(cfg.get("use_rate_priors", True))
        use_global_priors = bool(cfg.get("use_global_priors", True))
        engine_mode = (mode or cfg.get("engine_mode") or "rate_times_likelihood").strip().lower()
        if engine_mode not in {"rate_only", "likelihood_only", "rate_times_likelihood"}:
            engine_mode = "rate_times_likelihood"

        # Optional global prior weights (e.g., PPP expected counts)
        if not use_global_priors:
            log_prior_weights = None
        logW = (
            np.array([log_prior_weights.get(L, 0.0) for L in labels], dtype=float)
            if log_prior_weights
            else np.zeros(len(labels))
        )

        # Compute per-row, per-label rate densities and add as diagnostics
        rates = np.zeros((len(out), len(labels)), dtype=float)
        # Optional: per-component fake rate diagnostics
        fake_component_names = None
        fake_components_matrix = None
        has_fake = "fake" in labels
        j_fake = labels.index("fake") if has_fake else -1
        fake_model = self.registry.model("fake") if has_fake else None
        expose_fake_components = has_fake and hasattr(fake_model, "rate_components")

        # First pass: compute label rates and (optionally) record fake components
        for i, (_, row) in enumerate(out.iterrows()):
            for j, L in enumerate(labels):
                model = self.registry.model(L)
                try:
                    r = float(model.rate_density(row, self.ctx))
                except Exception:
                    r = 1.0
                rates[i, j] = max(r, 0.0)
            # per-row fake components
            if expose_fake_components:
                try:
                    comps = fake_model.rate_components(row, self.ctx)  # dict[name->rate]
                    if fake_component_names is None:
                        fake_component_names = list(comps.keys())
                        fake_components_matrix = np.zeros((len(out), len(fake_component_names)), dtype=float)
                    # align by known names; ignore unexpected
                    for cj, name in enumerate(fake_component_names):
                        val = float(comps.get(name, 0.0))
                        fake_components_matrix[i, cj] = max(val, 0.0)
                except Exception:
                    pass

        # Evidence matrix (log)
        logZ = np.vstack([out.get(f"logZ_{L}", np.full(len(out), -np.inf)) for L in labels]).T  # (N, K)

        # Apply mode and rate prior toggles
        with np.errstate(divide='ignore'):
            logR = np.log(rates + EPS_LOG)
        if not use_rate_priors:
            # evidence-only requested via config: neutralize rate priors
            logR[:, :] = 0.0
            expose_fake_components = False
        # Engine mode overrides combination
        if engine_mode == "rate_only":
            logZ[:, :] = 0.0
        elif engine_mode == "likelihood_only":
            logR[:, :] = 0.0
        # else combined: keep both

        # Emit per-label rate diagnostics
        for j, L in enumerate(labels):
            out[f"rate_{L}"] = rates[:, j]
        # Emit per-component fake rate diagnostics if available
        if expose_fake_components and fake_component_names is not None and fake_components_matrix is not None:
            for cj, name in enumerate(fake_component_names):
                col = f"rate_fake_{name}"
                out[col] = fake_components_matrix[:, cj]

        # Combine log rate prior with data evidence
        # Optionally emit the global log prior weights used (one per label, same for all rows)
        if log_prior_weights is not None and use_global_priors:
            try:
                for j, L in enumerate(labels):
                    out[f"log_prior_weight_{L}"] = float(logW[j])
            except Exception:
                pass
        logP_unnorm = logZ + logR + logW[None, :]

        # log-softmax
        m = np.max(logP_unnorm, axis=1, keepdims=True)
        P = np.exp(logP_unnorm - m)
        denom = np.sum(P, axis=1, keepdims=True)
        # Avoid division by zero if all -inf
        denom = np.where(denom <= 0, 1.0, denom)
        P = P / denom

        for j, L in enumerate(labels):
            out[f"p_{L}"] = P[:, j]

        # Diagnostics: physical vs fake totals and prior odds (if fake present)
        if "fake" in labels:
            j_fake = labels.index("fake")
            rate_fake = rates[:, j_fake]
            rate_phys = np.sum(rates, axis=1) - rate_fake
            # Avoid division by zero
            prior_odds = np.where(rate_fake > 0, rate_phys / np.maximum(rate_fake, EPS_LOG), np.nan)
            out["rate_fake_total"] = rate_fake
            out["rate_phys_total"] = rate_phys
            out["prior_odds_phys_over_fake"] = prior_odds
        else:
            out["rate_phys_total"] = np.sum(rates, axis=1)

        # Record which priors/mode were applied
        out["use_rate_priors"] = bool(use_rate_priors)
        out["use_global_priors"] = bool(use_global_priors)
        out["engine_mode"] = engine_mode

        # If a calibrated/used fake rate is present in context, record it for traceability
        try:
            cfg = getattr(self.ctx, "config", {}) or {}
            rho = cfg.get("fake_rate_rho_used", cfg.get("fake_rate_per_sr_per_A", None))
            if rho is not None:
                out["rho_used"] = float(rho)
        except Exception:
            pass

        return out
