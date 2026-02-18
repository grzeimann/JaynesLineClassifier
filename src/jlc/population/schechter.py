import numpy as np


class SchechterLF:
    def __init__(self, log10_Lstar: float, alpha: float, log10_phistar: float, Lmin: float | None = None, Lmax: float | None = None):
        self.log10_Lstar = float(log10_Lstar)
        self.alpha = float(alpha)
        self.log10_phistar = float(log10_phistar)
        # Optional luminosity bounds (in same units as L and L*)
        self.Lmin = float(Lmin) if Lmin is not None else None
        self.Lmax = float(Lmax) if Lmax is not None else None

    def phi(self, L: np.ndarray) -> np.ndarray:
        """Return number density per luminosity with optional luminosity bounds.

        Parameters
        - L: luminosity array (same units as implied by L*).

        Returns
        - phi(L) with zeros outside [Lmin, Lmax] if bounds are set.
        """
        L = np.asarray(L, dtype=float)
        Lstar = 10 ** self.log10_Lstar
        phistar = 10 ** self.log10_phistar
        # Base Schechter form
        x = np.where(Lstar > 0, L / Lstar, 0.0)
        phi = (phistar / max(Lstar, 1e-300)) * (np.power(x, self.alpha, where=x>0) * np.exp(-x))
        # Apply luminosity bounds if provided
        if self.Lmin is not None or self.Lmax is not None:
            mask = np.ones_like(L, dtype=bool)
            if self.Lmin is not None:
                mask &= (L >= self.Lmin)
            if self.Lmax is not None:
                mask &= (L <= self.Lmax)
            phi = np.where(mask, phi, 0.0)
        # Ensure non-negative and finite
        phi = np.where(np.isfinite(phi) & (phi >= 0), phi, 0.0)
        return phi
