import numpy as np


class SelectionModel:
    def completeness(self, F: np.ndarray, wave_obs: float) -> np.ndarray:
        """Return selection completeness in [0,1] for each flux value at given wave.
        Default stub returns ones (no selection losses)."""
        F = np.asarray(F, dtype=float)
        return np.ones_like(F)
