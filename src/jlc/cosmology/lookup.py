import numpy as np
from typing import Optional

from astropy import units as u
from astropy.cosmology import Cosmology, Planck18


class AstropyCosmology:
    """Cosmology backed by astropy.cosmology.

    Provides:
      - luminosity_distance(z): luminosity distance in Mpc (float)
      - dV_dz(z): differential comoving volume per unit redshift per steradian
        in Mpc^3 / sr (float)

    For speed, can optionally use a tabulated interpolation over a predefined
    redshift grid within [zmin, zmax]. Outside this range, exact astropy values
    are used. API remains scalar-focused but accepts numpy arrays too.
    """

    def __init__(
        self,
        cosmo: Optional[Cosmology] = None,
        use_interpolated: bool = True,
        zmin: float = 0.0,
        zmax: float = 6.0,
        grid_size: int = 2000,
    ):
        self.cosmo: Cosmology = cosmo if cosmo is not None else Planck18
        self.use_interp = bool(use_interpolated)
        self.zmin = float(zmin)
        self.zmax = float(zmax)
        self.grid_size = int(grid_size)

        self._zgrid = None
        self._dL_grid = None
        self._dVdz_grid = None

        if self.use_interp and self.grid_size > 1 and self.zmax > self.zmin:
            self._build_lookup()

    def _build_lookup(self):
        z = np.linspace(self.zmin, self.zmax, self.grid_size)
        # Ensure strictly positive minimum redshift to avoid log(0) users might do
        z = np.maximum(z, 1e-8)
        dL = self.cosmo.luminosity_distance(z).to_value(u.Mpc)
        dV_dz = self.cosmo.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr)
        self._zgrid = z
        self._dL_grid = dL
        self._dVdz_grid = dV_dz

    def _interp_or_exact(self, z, ygrid):
        # Helper that linearly interpolates within range; else computes exact
        if np.isscalar(z):
            z = float(max(z, 1e-8))
            if self._zgrid is not None and self.zmin <= z <= self.zmax:
                return float(np.interp(z, self._zgrid, ygrid))
            # Fallback to exact
            return None
        # Vector path
        z = np.asarray(z, dtype=float)
        z = np.maximum(z, 1e-8)
        if self._zgrid is None:
            return None
        mask = (z >= self.zmin) & (z <= self.zmax)
        out = np.empty_like(z, dtype=float)
        out[mask] = np.interp(z[mask], self._zgrid, ygrid)
        # Mark out-of-range with NaN to trigger exact path by caller per element
        out[~mask] = np.nan
        return out

    def luminosity_distance(self, z):
        """Luminosity distance D_L in Mpc.
        Accepts scalar or array; returns float or ndarray of float.
        """
        if self._dL_grid is not None:
            y = self._interp_or_exact(z, self._dL_grid)
            if y is not None:
                if np.isscalar(z):
                    if np.isnan(y):
                        pass  # compute exact below
                    else:
                        return float(y)
                else:
                    # For arrays, fill NaNs with exact values
                    z_arr = np.asarray(z, dtype=float)
                    need = np.isnan(y)
                    if np.any(need):
                        y[need] = self.cosmo.luminosity_distance(z_arr[need]).to_value(u.Mpc)
                    return y
        # Exact path
        z = np.asarray(z, dtype=float)
        z = np.maximum(z, 1e-8)
        dl = self.cosmo.luminosity_distance(z).to_value(u.Mpc)
        return float(dl) if dl.shape == () else dl

    def dV_dz(self, z):
        """Differential comoving volume per unit redshift per steradian in Mpc^3/sr.
        Accepts scalar or array; returns float or ndarray of float.
        """
        if self._dVdz_grid is not None:
            y = self._interp_or_exact(z, self._dVdz_grid)
            if y is not None:
                if np.isscalar(z):
                    if np.isnan(y):
                        pass
                    else:
                        return float(y)
                else:
                    z_arr = np.asarray(z, dtype=float)
                    need = np.isnan(y)
                    if np.any(need):
                        y[need] = self.cosmo.differential_comoving_volume(z_arr[need]).to_value(u.Mpc**3 / u.sr)
                    return y
        # Exact path
        z = np.asarray(z, dtype=float)
        z = np.maximum(z, 1e-8)
        dv = self.cosmo.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr)
        return float(dv) if dv.shape == () else dv
