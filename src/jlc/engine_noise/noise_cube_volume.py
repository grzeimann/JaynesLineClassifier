from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
from .noise_cube_model import NoiseCube


def bounds_from_noise_cube(cube: NoiseCube) -> Dict[str, Any]:
    """Return RA/Dec/Wave bounds and steps from cube grids.

    Returns dict with keys: ra_min, ra_max, dec_min, dec_max, wave_min, wave_max,
    n_ra, n_dec, n_wave, d_ra, d_dec, d_wave.
    """
    ra = np.asarray(cube.ra_grid, dtype=float)
    dec = np.asarray(cube.dec_grid, dtype=float)
    wave = np.asarray(cube.wave_grid, dtype=float)
    def step(g: np.ndarray) -> float:
        if g.size < 2:
            return np.nan
        return float(np.median(np.diff(g)))
    return {
        'ra_min': float(np.nanmin(ra)),
        'ra_max': float(np.nanmax(ra)),
        'dec_min': float(np.nanmin(dec)),
        'dec_max': float(np.nanmax(dec)),
        'wave_min': float(np.nanmin(wave)),
        'wave_max': float(np.nanmax(wave)),
        'n_ra': int(ra.size),
        'n_dec': int(dec.size),
        'n_wave': int(wave.size),
        'd_ra': step(ra),
        'd_dec': step(dec),
        'd_wave': step(wave),
    }


def simulation_volume_from_noise_cube(cube: NoiseCube) -> Dict[str, Any]:
    """Compute simple voxel-count based geometric volume for valid voxels.

    Returns dict with:
    - n_vox_total, n_vox_valid, frac_valid
    - bounds: output of bounds_from_noise_cube
    """
    m = cube.mask
    if m is None:
        n_valid = int(np.count_nonzero(np.isfinite(cube.noise) & (cube.noise > 0)))
        n_total = int(np.prod(cube.noise.shape))
    else:
        n_total = int(np.prod(m.shape))
        n_valid = int(n_total - int(np.count_nonzero(m)))
    bounds = bounds_from_noise_cube(cube)
    return {
        'n_vox_total': n_total,
        'n_vox_valid': n_valid,
        'frac_valid': (float(n_valid) / float(n_total) if n_total > 0 else np.nan),
        'bounds': bounds,
    }
