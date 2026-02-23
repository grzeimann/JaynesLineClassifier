import numpy as np
import pandas as pd
from jlc.utils.constants import EPS_LOG


class JaynesianEngine:
    def __init__(self, registry, ctx):
        self.registry = registry
        self.ctx = ctx

    @classmethod
    def from_config(cls, cfg: dict, ctx):
        """
        Construct an engine from a configuration dictionary.

        Expected minimal schema:
        {
            "labels": [
                {"label": "lae", "hyperparams": {...}},
                {"label": "oii", "hyperparams": {...}},
                {"label": "fake", "hyperparams": {...}},
            ]
        }
        Any additional keys inside the per-label dict are forwarded to the
        corresponding label constructor if present (e.g., rest_wave ignored
        for Fake).
        """
        from jlc.labels.registry import LabelRegistry as _LabelRegistry
        # Lazy imports to avoid circular dependencies at import time
        from jlc.labels.lae import LAELabel as _LAE
        from jlc.labels.oii import OIILabel as _OII
        from jlc.labels.fake import FakeLabel as _Fake
        CLASS_MAP = {
            "lae": _LAE,
            "oii": _OII,
            "fake": _Fake,
        }
        labels_cfg = (cfg or {}).get("labels", [])
        models = []
        for item in labels_cfg:
            try:
                name = str(item.get("label")).lower()
            except Exception:
                continue
            cls_label = CLASS_MAP.get(name)
            if cls_label is None:
                continue
            # Build with standardized constructor; pass through attachments from ctx when possible
            hp = item.get("hyperparams", None)
            try:
                m = cls_label(
                    hyperparams=hp,
                    cosmology=getattr(ctx, "cosmo", None),
                    selection_model=getattr(ctx, "selection", None),
                    flux_grid=getattr(getattr(ctx, "caches", {}), "get", lambda k, d=None: None)("flux_grid") if hasattr(getattr(ctx, "caches", {}), "get") else getattr(getattr(ctx, "caches", None), "flux_grid", None),
                    measurement_modules=item.get("measurement_modules", None),
                )
            except TypeError:
                # Fall back to minimal
                m = cls_label(hyperparams=hp)
            models.append(m)
        registry = _LabelRegistry(models)
        return cls(registry, ctx)

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

    def export_model_state(self) -> dict:
        """
        Export a snapshot of the current model state for all labels.
        Uses each label's to_config() helper so hyperparameters and minimal
        identifiers are serializable.
        """
        state = {}
        for name in self.registry.labels:
            try:
                m = self.registry.model(name)
                state[name] = dict(m.to_config()) if hasattr(m, "to_config") else dict(hyperparams=m.get_hyperparams_dict())
            except Exception:
                state[name] = {}
        return state

    def run_hierarchical_update(self, df: pd.DataFrame | None = None) -> dict:
        """
        Run per-label hierarchical updates using current posteriors as weights.

        If df is None or missing posterior columns p_<label>, we compute
        posteriors first using compute_posteriors.

        Returns a dict mapping label name to updated hyperparameters (as dicts).
        """
        if df is None:
            df = pd.DataFrame()
        need_post = False
        for L in self.registry.labels:
            if f"p_{L}" not in df.columns:
                need_post = True
                break
        if need_post or df.empty:
            # If no data provided, create an empty DataFrame to avoid crashes
            if df is None or df.empty:
                df_comp = pd.DataFrame({"wave_obs": [], "flux_hat": [], "flux_err": []})
            else:
                df_comp = df
            df = self.compute_posteriors(df_comp)
        # Dispatch to labels
        for L in self.registry.labels:
            m = self.registry.model(L)
            try:
                w = df.get(f"p_{L}")
                if w is None:
                    # fallback: uniform tiny weights
                    w = pd.Series(np.zeros(len(df), dtype=float))
                m.update_hyperparams(df, w, self.ctx)
            except Exception:
                # keep going even if a label declines to update
                pass
        return {L: self.registry.model(L).get_hyperparams_dict() for L in self.registry.labels}

    def simulate_from_model(self, n_per_label: int | dict | None = None, **kwargs) -> pd.DataFrame:
        """
        Generate a synthetic catalog by delegating to each label's simulator.

        Parameters
        ----------
        n_per_label : int | dict | None
            If int, acts as a soft cap via rng on each label's PPP when supported
            (labels may ignore and use their own μ). If dict, maps label->kwargs
            overrides including 'n' or other simulator options.
        **kwargs : forwarded to label.simulate_catalog where applicable.

        Returns
        -------
        DataFrame with unified schema across labels.
        """
        dfs = []
        for L in self.registry.labels:
            m = self.registry.model(L)
            lab_kwargs = dict(kwargs)
            if isinstance(n_per_label, dict) and L in n_per_label and isinstance(n_per_label[L], dict):
                lab_kwargs.update(n_per_label[L])
            try:
                df_lab, _meta = m.simulate_catalog(ctx=self.ctx, **lab_kwargs)
            except TypeError:
                # older signature without keywords
                try:
                    df_lab, _meta = m.simulate_catalog(self.ctx, **lab_kwargs)
                except Exception:
                    continue
            except Exception:
                continue
            if df_lab is not None and not df_lab.empty:
                dfs.append(df_lab)
        if len(dfs) == 0:
            return pd.DataFrame(columns=["ra","dec","true_class","wave_obs","flux_true","flux_hat","flux_err"]) 
        return pd.concat(dfs, ignore_index=True)

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

        # Compute per-row, per-label rate densities and add as diagnostics
        rates = np.zeros((len(out), len(labels)), dtype=float)

        # First pass: compute label rates and (optionally) record fake components
        for i, (_, row) in enumerate(out.iterrows()):
            for j, L in enumerate(labels):
                model = self.registry.model(L)
                try:
                    r = float(model.rate_density(row, self.ctx))
                except Exception:
                    r = 1.0
                rates[i, j] = max(r, 0.0)


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

        # Combine log rate prior with data evidence
        logP_unnorm = logZ + logR

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
