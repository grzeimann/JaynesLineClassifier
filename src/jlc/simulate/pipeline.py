from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Sequence, Optional
import numpy as np

from jlc.engine_noise.noise_cube_model import NoiseCube


@dataclass
class NoiseCubeReader:
    """Lightweight wrapper to read 2D noise slices from a NoiseCube.

    The NoiseCube stores data with axes ordered as (n_ra, n_dec, n_wave).
    """
    cube: NoiseCube

    @property
    def n_lambda(self) -> int:
        return int(self.cube.wave_grid.size)

    def read_noise_slice(self, k: int) -> np.ndarray:
        """Return a 2D (n_ra, n_dec) slice at wavelength index k.

        Invalid voxels are set to np.nan for convenient masking downstream.
        """
        if k < 0 or k >= self.n_lambda:
            raise IndexError(f"lambda index k={k} out of bounds [0,{self.n_lambda-1}]")
        sl = np.asarray(self.cube.noise[:, :, k], dtype=float)
        # Apply mask if present
        if self.cube.mask is not None:
            m = np.asarray(self.cube.mask[:, :, k], dtype=bool)
            sl = np.where(m, np.nan, sl)
        # Auto-mask non-finite or <= 0 as invalid
        sl = np.where(~np.isfinite(sl) | (sl <= 0), np.nan, sl)
        return sl


class LambdaSliceSpaxelIndex:
    """Index spaxels of a single wavelength slice by noise bins.

    Parameters
    ----------
    noise_slice_2d : np.ndarray
        2D array of shape (n_ra, n_dec) with noise values; invalid should be NaN.
    noise_bin_edges : Sequence[float]
        Monotonic array of bin edges for noise histogramming.
    """

    def __init__(self, noise_slice_2d: np.ndarray, noise_bin_edges: Sequence[float]):
        self.noise = np.asarray(noise_slice_2d, dtype=float)
        self.noise_bin_edges = np.asarray(noise_bin_edges, dtype=float)
        if self.noise.ndim != 2:
            raise ValueError("noise_slice_2d must be 2D (n_ra, n_dec)")
        if self.noise_bin_edges.size < 2:
            raise ValueError("noise_bin_edges must have at least 2 elements")
        # Build flat index lists per noise bin for efficient sampling
        self._flat_indices_per_bin: list[np.ndarray] = []
        flat = self.noise.ravel()
        # Compute bin index per valid pixel
        valid = np.isfinite(flat)
        bin_ids = np.full(flat.shape, -1, dtype=int)
        if np.any(valid):
            # digitize returns indices in 1..len(edges)-1; convert to 0-based
            bin_ids_valid = np.digitize(flat[valid], self.noise_bin_edges) - 1
            # clamp to range [-1, n_bins-1]
            n_bins = self.noise_bin_edges.size - 1
            bin_ids_valid = np.where((bin_ids_valid >= 0) & (bin_ids_valid < n_bins), bin_ids_valid, -1)
            bin_ids[valid] = bin_ids_valid
        # Collect indices per bin
        n_bins = self.noise_bin_edges.size - 1
        for j in range(n_bins):
            idx = np.nonzero(bin_ids == j)[0]
            self._flat_indices_per_bin.append(idx)
        self.shape = self.noise.shape

    @property
    def n_bins(self) -> int:
        return len(self._flat_indices_per_bin)

    def noise_bin_center(self, j: int) -> float:
        e = self.noise_bin_edges
        if j < 0 or j >= e.size - 1:
            raise IndexError("noise bin j out of range")
        return 0.5 * (float(e[j]) + float(e[j+1]))

    def sample_spaxels(self, j: int, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n spaxel coordinates (ira, idec) from bin j with replacement.

        Falls back to sampling from all valid spaxels if bin j is empty.
        If no valid spaxels exist at all, returns empty arrays.
        """
        if n <= 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        idxs = self._flat_indices_per_bin[j] if (0 <= j < self.n_bins) else np.array([], dtype=int)
        if idxs.size == 0:
            # fallback: any valid pixel
            valid = np.nonzero(np.isfinite(self.noise.ravel()))[0]
            idxs = valid
        if idxs.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        # Sample with replacement for simplicity
        take = rng.integers(0, idxs.size, size=int(n))
        flat_sel = idxs[take]
        # Convert flat indices to (ira, idec)
        ira, idec = np.unravel_index(flat_sel, self.shape)
        return ira.astype(int, copy=False), idec.astype(int, copy=False)


def wcs_from_indices(cube: NoiseCube, ira: Sequence[int], idec: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Map spaxel indices to (ra, dec) world coordinates using cube grids.

    Assumes indices refer to the (n_ra, n_dec) axes ordering of the cube.
    """
    ira = np.asarray(ira, dtype=int)
    idec = np.asarray(idec, dtype=int)
    ira = np.clip(ira, 0, cube.ra_grid.size - 1)
    idec = np.clip(idec, 0, cube.dec_grid.size - 1)
    ra = cube.ra_grid[ira]
    dec = cube.dec_grid[idec]
    return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)


def build_catalog_table(records: Sequence[dict]) -> 'np.recarray':
    """Convert list of dicts into a structured numpy record array.

    We return a recarray to avoid adding a hard pandas dependency for the
    pipeline stage; callers may convert to DataFrame as needed.
    """
    if len(records) == 0:
        # Return an empty recarray with core schema so downstream field checks pass
        core_fields = [
            ("ra", "f8"), ("dec", "f8"), ("lambda", "f8"),
            ("F_true", "f8"), ("F_fit", "f8"), ("F_error", "f8"),
            ("signal", "f8"), ("noise", "f8"), ("label", "O"),
        ]
        return np.recarray((0,), dtype=core_fields)
    # Determine fields and dtypes conservatively
    keys = sorted({k for r in records for k in r.keys()})
    dtype = []
    for k in keys:
        # Try to infer numeric vs string
        v0 = None
        for r in records:
            if k in r:
                v0 = r[k]
                break
        if isinstance(v0, (int, np.integer)):
            dtype.append((k, 'f8'))
        elif isinstance(v0, (float, np.floating)):
            dtype.append((k, 'f8'))
        else:
            dtype.append((k, 'O'))
    arr = np.recarray((len(records),), dtype=dtype)
    for i, r in enumerate(records):
        for k in keys:
            arr[i][k] = r.get(k)
    return arr
