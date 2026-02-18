import numpy as np


class SimpleCosmology:
    """Very rough cosmology stub for skeleton purposes.

    Provides luminosity_distance(z) and dV_dz(z) positive finite functions.
    Units are arbitrary but consistent for priors.
    """

    def __init__(self, H0: float = 70.0, c_km_s: float = 299792.458):
        self.H0 = float(H0)
        self.c = float(c_km_s)
        self._DHub = self.c / self.H0  # Mpc

    def luminosity_distance(self, z: float) -> float:
        # Simple approximation: D_L ≈ z * D_H (valid for small z), ensure positive
        z = float(max(z, 1e-6))
        return z * self._DHub

    def dV_dz(self, z: float) -> float:
        # Crude monotonic function behaving like z^2 for small z, scaled by D_H^3
        z = float(max(z, 1e-6))
        return (self._DHub ** 3) * (z ** 2)  # arbitrary scaling
