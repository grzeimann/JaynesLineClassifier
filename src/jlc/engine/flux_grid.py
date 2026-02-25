import numpy as np
import pandas as pd
from jlc.utils.constants import FLUX_MIN_FLOOR, THRESH_FACTOR_LOW, THRESH_FACTOR_HIGH, EPS_LOG


class FluxGrid:
    """Build a per-row flux grid and provide stable log-sum-exp integration.

    For Phase 1 we use a simple fixed log-spaced grid of positive fluxes.

    Guards:
    - Ensures strictly positive bounds and at least modest dynamic range.
    - Provides ensure_threshold(thr, ...) to expand the grid around a selection threshold.
    - Optional per-row windowing around the measured flux: restrict integration
      to [F_hat − k·σ, F_hat + k·σ] with k = window_sigma to accelerate inference.

    Runtime statistics (for timing detail diagnostics):
    - stats_calls: number of times grid(row) was invoked
    - stats_points_total: total number of points returned across all calls
    Use these to compute the average per-row flux grid size actually used.

    Use reset_stats() to zero the counters at the start of a timed section so the
    reported averages reflect only that section.
    """

    def __init__(self, Fmin: float = 1e-18, Fmax: float = 1e-14, n: int = 128,
                 *, window_sigma: float | None = None, window_min_n: int = 16):
        from jlc.utils.logging import log as _log
        self.Fmin = float(Fmin)
        self.Fmax = float(Fmax)
        self.n = int(n)
        # Optional per-row windowing params
        try:
            self.window_sigma = float(window_sigma) if window_sigma is not None else None
        except Exception:
            self.window_sigma = None
        try:
            self.window_min_n = int(window_min_n)
        except Exception:
            self.window_min_n = 16
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
        # Runtime stats for timing diagnostics
        self.stats_calls = 0
        self.stats_points_total = 0
        # Warn if dynamic range is too narrow (<~ 1 dex)
        try:
            rng_dex = np.log10(self.Fmax) - np.log10(self.Fmin)
            if rng_dex < 1.0:
                _log(f"[jlc.fluxgrid] Warning: flux grid spans only {rng_dex:.2f} dex; consider expanding around selection thresholds")
        except Exception:
            pass

    def reset_stats(self) -> None:
        """Reset runtime statistics counters for timing-detail diagnostics.

        Call this at the start of a measured section so averages refer only to
        the subsequent operations.
        """
        try:
            self.stats_calls = 0
            self.stats_points_total = 0
        except Exception:
            # Be tolerant if attributes were removed
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
        """Return (F_grid, log_w) for this row.

        If window_sigma is set and row provides valid flux_hat and flux_err>0,
        restrict the grid to [F_hat − k·σ, F_hat + k·σ] where k=window_sigma.
        Ensure at least window_min_n points by expanding to nearest neighbors.
        Fallback to full grid in degenerate cases.
        """
        # Default: full grid
        F = self._F_grid
        W = self._log_w
        k = self.window_sigma
        if k is None:
            Fout, Wout = F.copy(), W.copy()
            # update stats
            try:
                self.stats_calls += 1
                self.stats_points_total += Fout.size
            except Exception:
                pass
            return Fout, Wout
        try:
            F_hat = float(row.get("flux_hat")) if isinstance(row, pd.Series) else float("nan")
            sigma = float(row.get("flux_err")) if isinstance(row, pd.Series) else float("nan")
        except Exception:
            F_hat, sigma = float("nan"), float("nan")
        if not (np.isfinite(F_hat) and np.isfinite(sigma) and sigma > 0 and F_hat >= 0):
            Fout, Wout = F.copy(), W.copy()
            try:
                self.stats_calls += 1
                self.stats_points_total += Fout.size
            except Exception:
                pass
            return Fout, Wout
        lo = max(self.Fmin, F_hat - k * sigma)
        hi = min(self.Fmax, F_hat + k * sigma)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            Fout, Wout = F.copy(), W.copy()
            try:
                self.stats_calls += 1
                self.stats_points_total += Fout.size
            except Exception:
                pass
            return Fout, Wout
        mask = (F >= lo) & (F <= hi)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            # pick nearest index to F_hat
            j0 = int(np.argmin(np.abs(F - F_hat)))
            i0 = max(0, j0 - self.window_min_n // 2)
            i1 = min(F.size, i0 + self.window_min_n)
            sel = np.arange(i0, i1)
            Fout, Wout = F[sel].copy(), W[sel].copy()
            try:
                self.stats_calls += 1
                self.stats_points_total += Fout.size
            except Exception:
                pass
            return Fout, Wout
        # If too few points, expand symmetrically around center index
        if idx.size < self.window_min_n:
            j_center = int(np.argmin(np.abs(F - F_hat)))
            half = max(self.window_min_n // 2, 1)
            i0 = max(0, j_center - half)
            i1 = min(F.size, j_center + half + (self.window_min_n % 2))
            sel = np.arange(i0, i1)
            Fout, Wout = F[sel].copy(), W[sel].copy()
            try:
                self.stats_calls += 1
                self.stats_points_total += Fout.size
            except Exception:
                pass
            return Fout, Wout
        Fout, Wout = F[idx].copy(), W[idx].copy()
        try:
            self.stats_calls += 1
            self.stats_points_total += Fout.size
        except Exception:
            pass
        return Fout, Wout

    @staticmethod
    def logsumexp(a: np.ndarray) -> float:
        m = np.max(a)
        if not np.isfinite(m):
            return -np.inf
        return float(m + np.log(np.sum(np.exp(a - m))))
