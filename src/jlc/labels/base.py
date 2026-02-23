from __future__ import annotations
from abc import ABC
from typing import Sequence
import warnings
import pandas as pd
from jlc.types import EvidenceResult


class LabelModel(ABC):
    """Abstract base for per-label models used by the engine and simulator.

    Responsibilities per refactor plan:
    - rate_density(row, ctx): observed-space PPP intensity λ_L(x) (includes LF/fake-rate, cosmology, selection, and search-measure multipliers) evaluated at the row’s wavelength; returned as a finite, non-negative float.
    - extra_log_likelihood(row, ctx): measurement-only log-likelihood terms not already encoded in rate_density (e.g., flux-marginalized evidence on a shared FluxGrid).
    - simulate_catalog(...): optional per-label simulator using the same ingredients as rate_density for alignment with inference.
    """
    label: str
    measurement_modules: Sequence

    # --- Hyperparameter helpers (generic defaults) ---
    def _coerce_hyperparams(self, hp):
        return hp  # subclass should override to enforce dataclass

    def get_hyperparams(self):
        return getattr(self, "hyperparams", None)

    def get_hyperparams_dict(self):
        import dataclasses
        hp = getattr(self, "hyperparams", None)
        try:
            return dataclasses.asdict(hp) if hp is not None else {}
        except Exception:
            return {}

    def set_hyperparams(self, **updates):
        hp_dict = self.get_hyperparams_dict()
        hp_dict.update(updates)
        cls = getattr(self, "hyperparam_cls", None)
        if cls is not None:
            try:
                self.hyperparams = cls(**hp_dict)
                return
            except Exception:
                pass
        # fallback: store dict
        self.hyperparams = hp_dict

    def to_config(self) -> dict:
        """Serialize minimal label configuration (hyperparameters only)."""
        return {
            "label": getattr(self, "label", None),
            "hyperparams": self.get_hyperparams_dict(),
        }

    def extra_log_likelihood(self, row: pd.Series, ctx) -> float:
        """Return measurement-only log-evidence not included in rate_density.

        Inputs
        ------
        row : pandas.Series
            A catalog row with at least wave_obs, flux_hat, flux_err and any
            additional fields required by measurement modules.
        ctx : SharedContext
            Shared runtime context providing cosmology, selection, caches, config.

        Returns
        -------
        float
            log evidence (can be -inf) marginalized over latent variables
            such as true flux using the shared FluxGrid in ctx.caches.

        Default returns 0.0. Subclasses should override.
        """
        return 0.0

    def rate_density(self, row: pd.Series, ctx) -> float:
        """Return observed-space rate density r(row) ≥ 0 (per sr per Å).

        This combines intrinsic LF/fake-rate, cosmology (dV/dz, distances),
        selection completeness, and any effective search-measure multiplier.

        Phase 1 default: return 1.0 to maintain backward compatibility.
        Subclasses should override for physically meaningful rates.
        """
        return 1.0

    def update_hyperparams(self, df: pd.DataFrame, weights: pd.Series, ctx) -> None:
        """Optional hierarchical update hook (no-op by default)."""
        return
