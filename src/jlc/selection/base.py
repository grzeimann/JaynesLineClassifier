from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any


# ==========================
# Noise and S/N completeness
# ==========================

@dataclass
class NoiseModel:
    """Noise model mapping (wave_true, RA, Dec, IFU) → sigma.

    Phase 1: flat sigma or 1D wavelength table. Spatial inputs are accepted for
    future extensions but ignored here.
    """
    wave_grid: Optional[np.ndarray] = None
    sigma_wave: Optional[np.ndarray] = None
    default_sigma: float = 1.0

    def sigma(self, wave_true: float, ra: float | None = None, dec: float | None = None, ifu_id: int | None = None) -> float:
        try:
            # Try wavelength-dependent lookup if configured
            if self.wave_grid is not None and self.sigma_wave is not None:
                idx = int(np.argmin(np.abs(np.asarray(self.wave_grid, dtype=float) - float(wave_true))))
                s = float(np.asarray(self.sigma_wave, dtype=float)[idx])
                return float(s if np.isfinite(s) and s > 0 else self.default_sigma)
        except Exception:
            pass
        return float(self.default_sigma)


class SNCompletenessModel(ABC):
    """Abstract base for completeness as a function of S/N_true and wavelength."""

    @abstractmethod
    def completeness(self, sn_true: float, wave_true: float, latent: Dict[str, Any]) -> float:  # pragma: no cover - interface
        raise NotImplementedError

@dataclass
class SNLogisticPerLambdaBin(SNCompletenessModel):
    """Logistic (technically tanh, but close enough) S/N completeness with wavelength-bin dependence.

    C = 0.5 * (1 + tanh((S/N_true - sn50_j)/width_j)) for λ in bin j.
    """
    wave_bins: np.ndarray
    sn50: np.ndarray
    width: np.ndarray

    def __post_init__(self) -> None:
        self.wave_bins = np.asarray(self.wave_bins, dtype=float)
        self.sn50 = np.asarray(self.sn50, dtype=float)
        self.width = np.asarray(self.width, dtype=float)
        if self.wave_bins.size != self.sn50.size + 1:
            raise ValueError("wave_bins must have length len(sn50)+1")
        if self.width.size != self.sn50.size:
            raise ValueError("width must have same length as sn50")

    def _bin_index(self, wave_true: float) -> int:
        j = int(np.searchsorted(self.wave_bins, float(wave_true), side="right") - 1)
        return int(np.clip(j, 0, self.sn50.size - 1))

    def completeness(self, sn_true: float, wave_true: float, latent: Dict[str, Any]) -> float:
        try:
            j = self._bin_index(wave_true)
            x = (float(sn_true) - float(self.sn50[j])) / float(self.width[j])
            c = 0.5 * (1.0 + np.tanh(x))
            if not np.isfinite(c):
                return 0.0
            return float(np.clip(c, 0.0, 1.0))
        except Exception:
            return 0.0


class SelectionModel:
    """Selection model supporting S/N-based completeness per label.

    New API:
    - completeness_sn(label_or_name, F_array, wave_true[, ra, dec, ifu_id]) → array in [0,1]
      Uses noise_model.sigma(wave_true, ...) to compute S/N_true = F/sigma and applies the
      per-label SNCompletenessModel if configured. Safe fallback to ones if ingredients are missing.

    Legacy shim paths for flux-threshold completeness have been removed from the codebase.
    """
    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        sn_models: Dict[str, SNCompletenessModel] | None = None,
        # Deprecated/ignored legacy knobs for backward compatibility:
        f_lim: float | None = None,
        F50: float | None = None,
        w: float | None = None,
        **kwargs,
    ):
        # Incremental redesign: optional noise model and S/N models mapping
        self.noise_model: NoiseModel | None = noise_model
        self._sn_models: Dict[str, SNCompletenessModel] = dict(sn_models or {})
        # Silently accept legacy knobs to avoid breaking older tests/callers.
        # They have no effect in the new S/N-based completeness path.
        self._deprecated_f_lim = f_lim
        self._deprecated_F50 = F50
        self._deprecated_w = w

    # --------- Incremental S/N-based completeness hooks (optional) ---------
    def set_noise_model(self, noise_model: NoiseModel | None) -> None:
        self.noise_model = noise_model

    def set_sn_model_for(self, label_name: str, model: SNCompletenessModel | None) -> None:
        if model is None:
            self._sn_models.pop(str(label_name), None)
        else:
            self._sn_models[str(label_name)] = model

    def sn_model_for_label(self, label_name: str) -> SNCompletenessModel | None:
        return self._sn_models.get(str(label_name))

    def completeness(self, label, latent: Dict[str, Any]) -> float:
        """Evaluate S/N-based completeness for a single latent using label object.

        Returns 1.0 if any ingredient is missing. Safe clamps to [0,1].
        """
        try:
            if self.noise_model is None:
                return 1.0
            lname = getattr(label, "label", str(label))
            model = self.sn_model_for_label(lname)
            if model is None:
                return 1.0
            F_true = float(latent.get("F_true", 0.0))
            wave_true = float(latent.get("wave_true", 0.0))
            if not (np.isfinite(F_true) and F_true > 0 and np.isfinite(wave_true) and wave_true > 0):
                return 0.0
            sigma = float(self.noise_model.sigma(wave_true, ra=latent.get("ra"), dec=latent.get("dec"), ifu_id=latent.get("ifu_id")))
            if not (np.isfinite(sigma) and sigma > 0):
                return 0.0
            sn_true = F_true / sigma
            c = float(model.completeness(sn_true, wave_true, latent))
            if not np.isfinite(c) or c < 0:
                return 0.0
            return float(np.clip(c, 0.0, 1.0))
        except Exception:
            return 1.0

    # New vectorized array API replacing legacy flux-threshold completeness
    def completeness_sn_array(
        self,
        label_or_name: Any,
        F_array: np.ndarray,
        wave_true: float,
        *,
        ra: float | None = None,
        dec: float | None = None,
        ifu_id: int | None = None,
    ) -> np.ndarray:
        """Return per-flux completeness values in [0,1] using S/N-based model.

        If noise_model or the label's SN model are not set, returns an array of ones.
        """
        try:
            F = np.asarray(F_array, dtype=float)
        except Exception:
            F = np.array(F_array, dtype=float)
        if F.size == 0:
            return np.asarray(F, dtype=float)
        # Missing pieces → neutral (ones)
        if self.noise_model is None:
            return np.ones_like(F, dtype=float)
        lname = getattr(label_or_name, "label", str(label_or_name))
        model = self.sn_model_for_label(lname)
        if model is None:
            return np.ones_like(F, dtype=float)
        try:
            sigma = float(self.noise_model.sigma(float(wave_true), ra=ra, dec=dec, ifu_id=ifu_id))
        except Exception:
            sigma = float("nan")
        if not (np.isfinite(sigma) and sigma > 0):
            return np.zeros_like(F, dtype=float)
        sn = np.where(sigma > 0, F / sigma, 0.0)
        # Evaluate per element in the given wavelength bin
        out = np.empty_like(F, dtype=float)
        for i, s in enumerate(np.ravel(sn)):
            try:
                c = float(model.completeness(float(s), float(wave_true), {"ra": ra, "dec": dec, "ifu_id": ifu_id}))
            except Exception:
                c = 1.0
            if not np.isfinite(c):
                c = 0.0
            out[i] = float(np.clip(c, 0.0, 1.0))
        return out.reshape(F.shape)

# -----------------------------
# Factory from PriorRecord
# -----------------------------

def build_selection_model_from_priors(record: Any, *, default_sigma: float = 1.0, label_name: str | None = None) -> SelectionModel | None:
    """Build a SelectionModel with NoiseModel + S/N completeness from a PriorRecord.

    Expected schema under record.hyperparams:
      selection:
        default_sigma: <float>  # optional
        sn:
          model: "logistic_per_lambda_bin"
          params:
            bins_wave: [...]
            sn50: [...]
            width: [...]

    Returns None if required blocks are missing. Callers can keep existing
    SelectionModel in that case. The returned SelectionModel includes the
    configured NoiseModel and an SN completeness entry bound to `label_name`
    (or record.label if label_name is None).
    """
    try:
        hp = getattr(record, "hyperparams", {}) or {}
        sel_hp = hp.get("selection", {}) or {}
        sn_hp = sel_hp.get("sn", {}) or {}
        model_name = str(sn_hp.get("model", "")).strip().lower()
        params = dict(sn_hp.get("params", {})) if isinstance(sn_hp.get("params", {}), dict) else {}
        # Build NoiseModel (default_sigma only for now)
        nm = NoiseModel(default_sigma=float(sel_hp.get("default_sigma", default_sigma)))
        # Build SN completeness model
        sn_model = None
        if model_name in ("logistic_per_lambda_bin", "sn_logistic_per_lambda_bin", "logistic"):
            bins = params.get("bins_wave") or params.get("wave_bins")
            sn50 = params.get("sn50")
            width = params.get("width")
            if bins is None or sn50 is None or width is None:
                return None
            sn_model = SNLogisticPerLambdaBin(np.asarray(bins, dtype=float), np.asarray(sn50, dtype=float), np.asarray(width, dtype=float))
        else:
            return None
        sel = SelectionModel(noise_model=nm, sn_models={})
        lname = label_name or getattr(record, "label", None) or "all"
        sel.set_sn_model_for(str(lname), sn_model)
        return sel
    except Exception:
        return None
