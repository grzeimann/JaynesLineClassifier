import numpy as np


class SchechterLF:
    def __init__(self, log10_Lstar: float, alpha: float, log10_phistar: float):
        self.log10_Lstar = float(log10_Lstar)
        self.alpha = float(alpha)
        self.log10_phistar = float(log10_phistar)

    def phi(self, L: np.ndarray) -> np.ndarray:
        """Return number density per luminosity (unnormalized units for skeleton)."""
        L = np.asarray(L, dtype=float)
        Lstar = 10 ** self.log10_Lstar
        phistar = 10 ** self.log10_phistar
        x = np.where(Lstar > 0, L / Lstar, 0.0)
        return (phistar / max(Lstar, 1e-300)) * (np.power(x, self.alpha, where=x>0) * np.exp(-x))
