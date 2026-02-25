from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Tuple, Any, Dict
import numpy as np
from astropy.io import fits

# Reuse the lightweight NoiseModel interface from selection.base
try:
    from jlc.selection.base import NoiseModel as BaseNoiseModel
except Exception:  # pragma: no cover - fallback for type checking
    class BaseNoiseModel:  # type: ignore
        def sigma(self, wave_true: float, ra: float | None = None, dec: float | None = None, ifu_id: int | None = None) -> float:
            return 1.0


@dataclass
class NoiseCube:
    """3D noise cube defined on a regular grid in (RA, Dec, wavelength).

    Attributes
    ----------
    noise : np.ndarray
        3D array with shape (n_ra, n_dec, n_wave).
    ra_grid, dec_grid, wave_grid : np.ndarray
        1D coordinate arrays defining the grid for each axis.
    mask : np.ndarray | None
        Optional boolean mask with same shape as noise. True marks invalid voxels.
    """
    noise: np.ndarray
    ra_grid: np.ndarray
    dec_grid: np.ndarray
    wave_grid: np.ndarray
    mask: np.ndarray | None = None

    @classmethod
    def from_fits(cls, path: str) -> "NoiseCube":
        """Load a NoiseCube from a FITS file using WCS info in the primary header only.

        Single implementation policy:
        - PrimaryHDU contains a 3D noise array (float).
        - The 1D RA/Dec/Wave grids are constructed exclusively from FITS WCS
          keywords in the primary header using CRVALn/CRPIXn/CDELTn and CTYPEn.
        - Axis identification is done by CTYPEn containing 'RA', 'DEC', and one of
          'WAVE'/'LAMBDA'/'WAVELENGTH' (case-insensitive).
        - Optional 'MASK' image extension (3D) is supported and transposed to match
          the (RA, DEC, WAVE) ordering.

        This matches the provided VDFI_COSMOS cube convention:
            CTYPE1='RA---TAN', CTYPE2='DEC--TAN', CTYPE3='WAVE',
            CUNIT1='deg', CUNIT2='deg', CUNIT3='Angstrom',
            CRPIX1=(N+1)/2, CRPIX2=(N+1)/2, CDELT3=2, CRPIX3=1, CRVAL3=3470.
        """
        def build_grid(n: int, crval: float, crpix: float, cdelt: float) -> np.ndarray:
            # FITS pixel coordinates are 1-based: world(x_i) = CRVAL + (i+1 - CRPIX)*CDELT
            i = np.arange(n, dtype=float)
            return crval + (i + 1.0 - crpix) * cdelt

        with fits.open(path, memmap=True) as hdul:
            phdu = hdul[0]
            data = np.asarray(phdu.data)
            if data is None:
                raise ValueError(f"Primary HDU has no data in {path}")
            hdr = phdu.header

            # Read axis definitions strictly from header (no HDU fallbacks)
            naxis = int(hdr.get('NAXIS', 0))
            if naxis < 3:
                raise ValueError(f"Expected 3D cube, found NAXIS={naxis} in {path}")

            # Collect header keywords for the first three axes (1..3)
            n_axes = [int(hdr.get(f'NAXIS{n}', 0)) for n in (1, 2, 3)]
            ctypes = [str(hdr.get(f'CTYPE{n}', '')).upper() for n in (1, 2, 3)]
            crvals = [float(hdr.get(f'CRVAL{n}')) for n in (1, 2, 3)]
            crpixs = [float(hdr.get(f'CRPIX{n}')) for n in (1, 2, 3)]
            cdeltas = [float(hdr.get(f'CDELT{n}')) for n in (1, 2, 3)]

            # Identify which FITS axis corresponds to RA/DEC/WAVE
            def find_axis(name_opts):
                for idx, c in enumerate(ctypes):
                    for opt in name_opts:
                        if opt in c:
                            return idx  # 0-based index for FITS axis (1->0, 2->1, 3->2)
                return None

            ra_ax = find_axis(['RA'])
            dec_ax = find_axis(['DEC'])
            wave_ax = find_axis(['WAVE', 'LAMBDA', 'WAVELENGTH'])
            if ra_ax is None or dec_ax is None or wave_ax is None:
                raise ValueError("Could not identify RA/DEC/WAVE axes from CTYPE1..3 header keywords")

            # Build grids for each identified axis
            ra = build_grid(n_axes[ra_ax], crvals[ra_ax], crpixs[ra_ax], cdeltas[ra_ax])
            dec = build_grid(n_axes[dec_ax], crvals[dec_ax], crpixs[dec_ax], cdeltas[dec_ax])
            wave = build_grid(n_axes[wave_ax], crvals[wave_ax], crpixs[wave_ax], cdeltas[wave_ax])

            # FITS axis order to numpy axis order for a 3D image: (3,2,1) -> (0,1,2)
            # Map FITS axis indices 0/1/2 to numpy axes 2/1/0 respectively
            fits_to_numpy = {0: 2, 1: 1, 2: 0}
            current_axes = [fits_to_numpy[ra_ax], fits_to_numpy[dec_ax], fits_to_numpy[wave_ax]]
            perm = tuple(current_axes)

            # Reorder the primary data so that resulting array axes are (RA, DEC, WAVE)
            noise = np.transpose(data, axes=perm)

            # Optional MASK extension: transpose to the same ordering
            mask = None
            if 'MASK' in hdul:
                raw_mask = np.asarray(hdul['MASK'].data)
                if raw_mask is not None and raw_mask.ndim == 3:
                    mask = np.transpose(raw_mask, axes=perm).astype(bool)

        cube = cls(noise=noise, ra_grid=np.asarray(ra), dec_grid=np.asarray(dec), wave_grid=np.asarray(wave), mask=mask)
        return cube.with_auto_mask()

    def with_auto_mask(self) -> "NoiseCube":
        """Return a copy with mask set to True where noise is invalid (<=0 or non-finite).
        Leaves any pre-existing mask combined with auto mask via logical OR.
        """
        auto = ~(np.isfinite(self.noise)) | (self.noise <= 0)
        if self.mask is None:
            m = auto
        else:
            m = np.asarray(self.mask, dtype=bool) | auto
        return replace(self, mask=m)

    def _nearest_index(self, grid: np.ndarray, x: float) -> int:
        # searchsorted gives insertion position; convert to nearest neighbor index
        g = np.asarray(grid, dtype=float)
        j = int(np.searchsorted(g, float(x), side='left'))
        if j <= 0:
            return 0
        if j >= g.size:
            return int(g.size - 1)
        # choose nearer of j-1 and j
        left = g[j-1]
        right = g[j]
        return int(j-1 if abs(x - left) <= abs(right - x) else j)

    def index_of(self, ra: float, dec: float, wave: float) -> Tuple[int, int, int]:
        ira = self._nearest_index(self.ra_grid, float(ra))
        idec = self._nearest_index(self.dec_grid, float(dec))
        ilam = self._nearest_index(self.wave_grid, float(wave))
        return ira, idec, ilam

    def value_at(self, ra: float, dec: float, wave: float, return_mask: bool = False):
        ira, idec, ilam = self.index_of(ra, dec, wave)
        val = float(self.noise[ira, idec, ilam])
        if self.mask is not None:
            m = bool(self.mask[ira, idec, ilam]) or (not np.isfinite(val)) or (val <= 0)
        else:
            m = (not np.isfinite(val)) or (val <= 0)
        if return_mask:
            return (np.nan if m else val), m
        return (np.nan if m else val)


class NoiseCubeModel(BaseNoiseModel):
    """NoiseModel backed by a 3D NoiseCube.

    If the targeted voxel is invalid/masked, sigma() returns np.nan.
    """

    def __init__(self, cube: NoiseCube, default_sigma: float = 1.0) -> None:
        self.cube = cube.with_auto_mask()
        self.default_sigma = float(default_sigma)

    def sigma(self, wave_true: float, ra: float | None = None, dec: float | None = None, ifu_id: int | None = None) -> float:  # type: ignore[override]
        if ra is None or dec is None:
            # Without sky position we cannot lookup 3D cube; be conservative and return default
            return float(self.default_sigma)
        val, m = self.cube.value_at(float(ra), float(dec), float(wave_true), return_mask=True)
        if m:
            return float('nan')
        try:
            v = float(val)
            if not np.isfinite(v) or v <= 0:
                return float('nan')
            return v
        except Exception:
            return float('nan')


def attach_noise_to_dataframe(cube: NoiseCube, df: Any, *, ra_col: str = 'ra', dec_col: str = 'dec', wave_col: str = 'wave', out_col: str = 'sigma_noise') -> Any:
    """Annotate a pandas DataFrame with nearest-neighbor cube noise values.

    Parameters
    ----------
    cube : NoiseCube
    df : pandas.DataFrame (duck-typed)
    ra_col, dec_col, wave_col : column names to read
    out_col : name of the output column to create

    Returns the modified DataFrame (same object) with a new column containing
    float noise values and NaN for invalid voxels. Uses simple chunking to
    bound memory.
    """
    import pandas as pd  # local import
    if not isinstance(df, pd.DataFrame):  # pragma: no cover - defensive
        raise TypeError("attach_noise_to_dataframe expects a pandas.DataFrame")

    ra_arr = df[ra_col].to_numpy(dtype=float, copy=False)
    dec_arr = df[dec_col].to_numpy(dtype=float, copy=False)
    wave_arr = df[wave_col].to_numpy(dtype=float, copy=False)

    n = ra_arr.size
    out = np.full(n, np.nan, dtype=float)
    # Chunk over rows to avoid pathological overhead (simple loop is OK for small n)
    chunk = max(1, int(10000))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        for i in range(start, stop):
            val = cube.value_at(ra_arr[i], dec_arr[i], wave_arr[i])
            out[i] = float(val) if np.isfinite(val) and val > 0 else np.nan
    df[out_col] = out
    return df
