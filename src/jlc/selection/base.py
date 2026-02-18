import numpy as np


class SelectionModel:
    def __init__(self, f_lim: float | None = None):
        """Selection model with optional hard flux threshold.

        If f_lim is provided, completeness is 1 where F > f_lim else 0.
        If None, completeness is 1 for all F (legacy behavior).
        """
        self.f_lim = float(f_lim) if f_lim is not None else None

    def completeness(self, F: np.ndarray, wave_obs: float) -> np.ndarray:
        """Return selection completeness in [0,1] for each flux value at given wave.
        Default: hard threshold if f_lim set, else ones.
        """
        F = np.asarray(F, dtype=float)
        if self.f_lim is None:
            return np.ones_like(F)
        return (F > self.f_lim).astype(float)
