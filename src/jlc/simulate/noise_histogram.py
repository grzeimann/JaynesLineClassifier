from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Sequence
import numpy as np

from jlc.engine_noise.noise_cube_model import NoiseCube
from .pipeline import NoiseCubeReader


@dataclass
class NoiseHistogram:
    """Histogram of noise per wavelength slice with survey area bookkeeping.

    Attributes
    ----------
    lambda_grid : np.ndarray
        1D wavelength grid (Angstrom) corresponding to cube.wave_grid.
    noise_bin_edges : np.ndarray
        1D array of noise bin edges (monotonic increasing), length = n_bins+1.
    counts : np.ndarray
        2D array of shape (n_lambda, n_bins) with counts of valid spaxels per bin.
    survey_area_sr : np.ndarray
        1D array of shape (n_lambda,) with total sky area (steradians) covered by
        valid spaxels in that wavelength slice.
    """

    lambda_grid: np.ndarray
    noise_bin_edges: np.ndarray
    counts: np.ndarray
    survey_area_sr: np.ndarray

    @property
    def n_lambda(self) -> int:
        return int(self.lambda_grid.size)

    @property
    def n_bins(self) -> int:
        return int(self.noise_bin_edges.size - 1)

    @property
    def noise_bin_centers(self) -> np.ndarray:
        e = np.asarray(self.noise_bin_edges, dtype=float)
        return 0.5 * (e[:-1] + e[1:])

    def _nearest_lambda_index(self, lam: float) -> int:
        g = np.asarray(self.lambda_grid, dtype=float)
        j = int(np.searchsorted(g, float(lam), side="left"))
        if j <= 0:
            return 0
        if j >= g.size:
            return int(g.size - 1)
        left = g[j-1]
        right = g[j]
        return int(j-1 if abs(lam - left) <= abs(right - lam) else j)

    def hist_at_lambda(self, lam: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return (centers, weights, k) for the nearest wavelength to lam.

        If total counts at slice k are zero, weights will be all zeros.
        """
        k = self._nearest_lambda_index(float(lam))
        centers = self.noise_bin_centers
        counts = np.asarray(self.counts[k], dtype=float)
        total = float(np.nansum(counts))
        if not np.isfinite(total) or total <= 0:
            weights = np.zeros_like(counts, dtype=float)
        else:
            weights = counts / total
        return centers, weights, k


def _compute_per_spaxel_area_sr(cube: NoiseCube) -> np.ndarray:
    """Compute sky area per spaxel (steradians) for each (ira, idec) position.

    Approximates the area of a pixel at (ra_i, dec_j) as:
        dΩ_ij ≈ ΔRA_rad * ΔDEC_rad * cos(dec_j)
    using central differences for grid spacings (falling back to edge differences).

    Returns
    -------
    area_2d : np.ndarray
        Array of shape (n_ra, n_dec) with per-spaxel steradian areas (>=0).
    """
    ra = np.asarray(cube.ra_grid, dtype=float)
    dec = np.asarray(cube.dec_grid, dtype=float)
    n_ra = ra.size
    n_dec = dec.size
    if n_ra < 2 or n_dec < 2:
        # Degenerate grid; return zeros to avoid dividing by zero later
        return np.zeros((n_ra, n_dec), dtype=float)
    # Spacings in radians
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    # Use median spacing as Δ for stability (RA assumed nearly uniform)
    dra = np.median(np.abs(np.diff(ra_rad)))
    # DEC spacing may be non-uniform; use median as well
    ddec = np.median(np.abs(np.diff(dec_rad)))
    # cos(dec) factor by row
    cos_dec = np.cos(dec_rad)
    # Broadcast to 2D (n_ra, n_dec)
    area_row = dra * ddec * cos_dec  # shape (n_dec,)
    area_2d = np.broadcast_to(area_row, (n_ra, n_dec)).astype(float, copy=True)
    # Ensure non-negative and finite
    area_2d = np.where(np.isfinite(area_2d) & (area_2d > 0), area_2d, 0.0)
    return area_2d


def build_noise_histogram(reader: NoiseCubeReader, noise_bin_edges: Sequence[float], *, enable_mem: bool = False) -> NoiseHistogram:
    """Build a NoiseHistogram by streaming slices from a NoiseCubeReader.

    Parameters
    ----------
    reader : NoiseCubeReader
        Streamer over 2D noise slices from a 3D NoiseCube.
    noise_bin_edges : Sequence[float]
        Histogram bin edges for noise values. Must be monotonic and length >= 2.

    Returns
    -------
    NoiseHistogram
        Persistent histogram object with counts per (λ, noise_bin) and survey area per λ.
    """
    from jlc.utils.logging import log as _log
    import time as _time

    e = np.asarray(noise_bin_edges, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("noise_bin_edges must be a 1D array with length >= 2")
    cube = reader.cube
    lam = np.asarray(cube.wave_grid, dtype=float)
    nlam = lam.size
    nbins = e.size - 1
    counts = np.zeros((nlam, nbins), dtype=np.int64)
    # Pre-compute per-spaxel area (steradian) from WCS grids
    area_2d = _compute_per_spaxel_area_sr(cube)
    survey_area_sr = np.zeros((nlam,), dtype=float)

    # Optional memory diagnostics
    if enable_mem:
        try:
            from .diagnostics import log_mem, array_nbytes
            log_mem("noise_hist:init", {"counts": array_nbytes(counts), "survey_area": array_nbytes(survey_area_sr), "area_2d": array_nbytes(area_2d)})
        except Exception:
            pass

    t0 = _time.time()
    # Progress interval: about 20 updates across the whole loop (at least every slice for tiny cubes)
    step = max(1, nlam // 20)

    for k in range(nlam):
        sl = reader.read_noise_slice(k)  # 2D float with NaN for invalid
        valid = np.isfinite(sl) & (sl > 0)
        if not np.any(valid):
            # Fully masked/invalid slice: leave zeros
            pass
        else:
            # Histogram counts for valid values
            h, _ = np.histogram(sl[valid], bins=e)
            counts[k, :] = h.astype(np.int64, copy=False)
            # Survey area: sum per-spaxel steradian area for valid spaxels
            try:
                survey_area_sr[k] = float(np.nansum(area_2d[valid]))
            except Exception:
                # Shape mismatch fallback (should not happen): compute from mean row area times valid count
                mean_area = float(np.nanmean(area_2d)) if np.any(np.isfinite(area_2d)) else 0.0
                survey_area_sr[k] = float(np.count_nonzero(valid)) * mean_area
        # Progress logging
        if (k % step == 0) or (k == nlam - 1):
            elapsed = _time.time() - t0
            done = k + 1
            rate = done / elapsed if elapsed > 0 else float('inf')
            remaining = nlam - done
            eta = remaining / rate if rate > 0 else float('inf')
            pct = (100.0 * done / nlam) if nlam > 0 else 100.0
            _log(f"[jlc.simulate] Building noise histogram: {done}/{nlam} slices ({pct:.0f}%), elapsed {elapsed:.1f}s, ETA {eta:.1f}s, {rate:.1f} slices/s")
            if enable_mem:
                try:
                    from .diagnostics import log_mem, array_nbytes
                    log_mem("noise_hist:progress", {"counts": array_nbytes(counts), "survey_area": array_nbytes(survey_area_sr)})
                except Exception:
                    pass

    total_elapsed = _time.time() - t0
    _log(f"[jlc.simulate] Built noise histogram for {nlam} λ-slices and {nbins} noise bins in {total_elapsed:.2f}s")
    if enable_mem:
        try:
            from .diagnostics import log_mem, array_nbytes
            log_mem("noise_hist:done", {"counts": array_nbytes(counts), "survey_area": array_nbytes(survey_area_sr)})
        except Exception:
            pass

    return NoiseHistogram(lambda_grid=lam, noise_bin_edges=e, counts=counts, survey_area_sr=survey_area_sr)
