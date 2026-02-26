from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any
import numpy as np

from jlc.selection.base import SelectionModel
from .diagnostics import get_completeness_tracer


class CompletenessProvider(Protocol):
    def completeness(self, F_true_grid: np.ndarray, lam: float, label: str) -> np.ndarray:
        """Return completeness C(F_true | lam, label) with shape like F_true_grid."""
        ...


@dataclass
class CatalogCompleteness:
    """Completeness for a specific catalog row with a known noise value.

    This bypasses the SelectionModel's NoiseModel lookup and directly queries the
    per-label SNCompletenessModel using S/N = F_true / noise.

    If noise <= 0 or non-finite, returns zeros.
    """
    selection: SelectionModel
    row: Any  # expects attributes/keys: noise, lambda (optional)

    def _get_noise(self) -> float:
        # Support both attribute and dict-like access
        try:
            n = getattr(self.row, "noise")
        except Exception:
            try:
                n = self.row["noise"]
            except Exception:
                n = np.nan
        return float(n)

    def completeness(self, F_true_grid: np.ndarray, lam: float, label: str) -> np.ndarray:
        F = np.asarray(F_true_grid, dtype=float)
        noise = float(self._get_noise())
        if not (np.isfinite(noise) and noise > 0):
            return np.zeros_like(F, dtype=float)
        # Pull SN model for label
        model = self.selection.sn_model_for_label(getattr(label, "label", str(label)))
        if model is None:
            # Missing SN model ⇒ neutral completeness (ones)
            return np.ones_like(F, dtype=float)
        sn = np.where(noise > 0, F / noise, 0.0)
        out = np.empty_like(F, dtype=float)
        tracer = get_completeness_tracer()
        for i, s in enumerate(np.ravel(sn)):
            try:
                c = float(model.completeness(float(s), float(lam), {"row": self.row}))
            except Exception:
                # Record provider exception and be conservative
                try:
                    tracer.on_exception("provider", str(label))
                except Exception:
                    pass
                c = 0.0
            if not np.isfinite(c) or c < 0:
                c = 0.0
            out[i] = float(np.clip(c, 0.0, 1.0))
        return out.reshape(F.shape)


@dataclass
class NoiseBinConditionedCompleteness:
    """Completeness conditioned on a fixed noise value (cell-level computations)."""
    selection: SelectionModel
    noise_value: float

    def completeness(self, F_true_grid: np.ndarray, lam: float, label: str) -> np.ndarray:
        F = np.asarray(F_true_grid, dtype=float)
        noise = float(self.noise_value)
        if not (np.isfinite(noise) and noise > 0):
            return np.zeros_like(F, dtype=float)
        model = self.selection.sn_model_for_label(getattr(label, "label", str(label)))
        if model is None:
            return np.ones_like(F, dtype=float)
        sn = np.where(noise > 0, F / noise, 0.0)
        out = np.empty_like(F, dtype=float)
        tracer = get_completeness_tracer()
        for i, s in enumerate(np.ravel(sn)):
            try:
                c = float(model.completeness(float(s), float(lam), {}))
            except Exception:
                # Record provider exception and be conservative
                try:
                    tracer.on_exception("provider", str(label))
                except Exception:
                    pass
                c = 0.0
            if not np.isfinite(c) or c < 0:
                c = 0.0
            out[i] = float(np.clip(c, 0.0, 1.0))
        return out.reshape(F.shape)


@dataclass
class NoiseHistogramCompleteness:
    """Completeness that marginalizes over a noise distribution per λ using a NoiseHistogram.

    weights/centers are taken from hist.hist_at_lambda(lam). Completeness is the
    weighted sum over noise-bin-conditioned completeness values.
    """
    selection: SelectionModel
    hist: Any  # expects .hist_at_lambda(lam) -> (centers, weights, k)

    def completeness(self, F_true_grid: np.ndarray, lam: float, label: str) -> np.ndarray:
        F = np.asarray(F_true_grid, dtype=float)
        centers, weights, _ = self.hist.hist_at_lambda(float(lam))
        # Guard: no weights or all zeros ⇒ zeros
        if centers is None or weights is None or len(weights) == 0 or np.all(np.asarray(weights) <= 0):
            return np.zeros_like(F, dtype=float)
        w = np.asarray(weights, dtype=float)
        csum = np.zeros_like(F, dtype=float)
        # Use per-bin conditioned completeness and weight-sum
        for n_val, wj in zip(np.asarray(centers, dtype=float), w):
            if not (np.isfinite(wj) and wj > 0):
                continue
            cj = NoiseBinConditionedCompleteness(selection=self.selection, noise_value=float(n_val))
            csum = csum + float(wj) * cj.completeness(F, lam, label)
        # Ensure within [0,1]
        csum = np.clip(csum, 0.0, 1.0)
        return csum
