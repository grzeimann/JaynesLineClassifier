import numpy as np
import pandas as pd


class FluxGrid:
    """Build a per-row flux grid and provide stable log-sum-exp integration.

    For Phase 1 we use a simple fixed log-spaced grid of positive fluxes.
    """

    def __init__(self, Fmin: float = 1e-20, Fmax: float = 1e-15, n: int = 64):
        self.Fmin = float(Fmin)
        self.Fmax = float(Fmax)
        self.n = int(n)
        # Precompute base grid and log-weights for trapezoidal rule in log-space
        self._F_grid = np.logspace(np.log10(self.Fmin), np.log10(self.Fmax), self.n)
        # Use simple uniform log-spacing weights
        dlogF = (np.log(self.Fmax) - np.log(self.Fmin)) / (self.n - 1)
        # Convert integral dF = F d(log F)
        self._log_w = np.log(self._F_grid) + np.log(dlogF)

    def grid(self, row: pd.Series):
        # In the skeleton we ignore row-dependence; could adapt bounds by flux_err later
        return self._F_grid.copy(), self._log_w.copy()

    @staticmethod
    def logsumexp(a: np.ndarray) -> float:
        m = np.max(a)
        if not np.isfinite(m):
            return -np.inf
        return float(m + np.log(np.sum(np.exp(a - m))))
