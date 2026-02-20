import numpy as np
import pandas as pd
from jlc.utils.constants import FLUX_MIN_FLOOR, THRESH_FACTOR_LOW, THRESH_FACTOR_HIGH, EPS_LOG


class FluxGrid:
    """Build a per-row flux grid and provide stable log-sum-exp integration.

    For Phase 1 we use a simple fixed log-spaced grid of positive fluxes.

    Guards:
    - Ensures strictly positive bounds and at least modest dynamic range.
    - Provides ensure_threshold(thr, ...) to expand the grid around a selection threshold.
    """

    def __init__(self, Fmin: float = 1e-18, Fmax: float = 1e-14, n: int = 128):
        from jlc.utils.logging import log as _log
        self.Fmin = float(Fmin)
        self.Fmax = float(Fmax)
        self.n = int(n)
        # Guards
        if not np.isfinite(self.Fmin) or self.Fmin <= 0:
            _log(f"[jlc.fluxgrid] Warning: invalid Fmin={self.Fmin}; clamping to {FLUX_MIN_FLOOR}")
            self.Fmin = FLUX_MIN_FLOOR
        if not np.isfinite(self.Fmax) or self.Fmax <= self.Fmin:
            new_max = self.Fmin * 1.0e4
            _log(f"[jlc.fluxgrid] Warning: invalid Fmax={Fmax}; setting to {new_max:.3e}")
            self.Fmax = new_max
        # Precompute base grid and log-weights for trapezoidal rule in log-space
        self._rebuild()
        # Warn if dynamic range is too narrow (<~ 1 dex)
        try:
            rng_dex = np.log10(self.Fmax) - np.log10(self.Fmin)
            if rng_dex < 1.0:
                _log(f"[jlc.fluxgrid] Warning: flux grid spans only {rng_dex:.2f} dex; consider expanding around selection thresholds")
        except Exception:
            pass

    def _rebuild(self):
        # Build grid and weights assuming strictly positive bounds
        self._F_grid = np.logspace(np.log10(self.Fmin), np.log10(self.Fmax), self.n)
        # Use simple uniform log-spacing weights
        dlogF = (np.log(self.Fmax) - np.log(self.Fmin)) / (max(self.n - 1, 1))
        # Convert integral dF = F d(log F)
        self._log_w = np.log(self._F_grid) + np.log(max(dlogF, EPS_LOG))

    def ensure_threshold(self, thr: float, factor_low: float = THRESH_FACTOR_LOW, factor_high: float = THRESH_FACTOR_HIGH) -> bool:
        """Ensure the grid comfortably straddles a given threshold thr.

        If expansion is required, rebuilds the grid and returns True; else False.
        """
        from jlc.utils.logging import log as _log
        changed = False
        try:
            thr = float(thr)
            if not (np.isfinite(thr) and thr > 0):
                return False
            target_min = max(thr * float(factor_low), FLUX_MIN_FLOOR)
            target_max = thr * float(factor_high)
            if self.Fmin > target_min:
                self.Fmin = target_min
                changed = True
            if self.Fmax < target_max:
                self.Fmax = target_max
                changed = True
            if changed:
                self._rebuild()
                _log(f"[jlc.fluxgrid] Expanded grid to cover threshold {thr:.3e}: Fmin={self.Fmin:.3e}, Fmax={self.Fmax:.3e}, n={self.n}")
            return changed
        except Exception:
            return False

    def grid(self, row: pd.Series):
        # In the skeleton we ignore row-dependence; could adapt bounds by flux_err later
        return self._F_grid.copy(), self._log_w.copy()

    @staticmethod
    def logsumexp(a: np.ndarray) -> float:
        m = np.max(a)
        if not np.isfinite(m):
            return -np.inf
        return float(m + np.log(np.sum(np.exp(a - m))))
