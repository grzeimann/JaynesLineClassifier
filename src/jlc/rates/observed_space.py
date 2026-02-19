"""
Observed-space rate utilities.

We express all rate densities in the same observed coordinates and measure:

  dV_obs = dλ * dF * dA_fiber

For Phase 1 we use an effective search measure function that can be scaled in
future milestones (e.g., number of fibers, exposure weighting, IFU masks). For
now it returns 1.0 unless configured otherwise via ctx.config.

Phase 2 adds helpers to build and evaluate an empirical fake λ-intensity shape
from virtual-volume detections. The PDF cache is stored under
ctx.caches["fake_lambda_pdf"].
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np


def calibrate_fake_rate_from_catalog(df: Any, ra_low: float, ra_high: float, dec_low: float, dec_high: float,
                                     wave_min: float, wave_max: float) -> float:
    """
    Estimate a homogeneous fake rate density ρ (per sr per Å) from a virtual catalog.

    Minimal estimator:
      ρ̂ = N / (Ω · Δλ)
    where N is the number of rows with wave_obs within [wave_min, wave_max],
    Ω is the solid angle of the rectangular sky box, and Δλ = wave_max − wave_min.

    Parameters
    ----------
    df : DataFrame-like (must support df.get("wave_obs").values)
    ra_low, ra_high, dec_low, dec_high : float
        Sky box bounds in degrees.
    wave_min, wave_max : float
        Wavelength band in Å.

    Returns
    -------
    float
        Estimated ρ̂ in units of 1/(sr·Å). Returns 0.0 if inputs invalid.
    """
    try:
        import numpy as _np
        from jlc.simulate.model_ppp import skybox_solid_angle_sr as _omega
        lam = _np.asarray(df.get("wave_obs")).astype(float)
        if lam.size == 0:
            return 0.0
        wmin = float(wave_min)
        wmax = float(wave_max)
        if not (_np.isfinite(wmin) and _np.isfinite(wmax)) or wmax <= wmin:
            return 0.0
        sel = _np.isfinite(lam) & (lam >= wmin) & (lam <= wmax)
        N = int(_np.sum(sel))
        Omega = float(_omega(ra_low, ra_high, dec_low, dec_high))
        Omega = max(Omega, 0.0)
        Dlam = float(wmax - wmin)
        if Omega <= 0.0 or Dlam <= 0.0:
            return 0.0
        rho_hat = float(N) / (Omega * Dlam)
        if not _np.isfinite(rho_hat) or rho_hat < 0:
            return 0.0
        return float(rho_hat)
    except Exception:
        return 0.0


def effective_search_measure(row: Any, ctx: Any) -> float:
    """
    Return the effective search measure multiplier to convert per-(sr·Å·flux)
    rates into per-candidate prior rate weight.

    Phase 2 behavior:
    - Pull simple, multiplicative knobs from ctx.config:
      - search_measure_scale (default 1.0)
      - n_fibers (default 1)
      - ifu_count (default 1)
      - exposure_scale (default 1.0)
    The returned multiplier is:
        scale = search_measure_scale * max(n_fibers,1) * max(ifu_count,1) * max(exposure_scale, 0)

    Notes:
    - Keep it dimensionless and modest to avoid destabilizing posteriors.
    - Future extensions can incorporate wavelength coverage actually explored for
      the candidate, masking, quality flags, etc.
    """
    try:
        cfg = getattr(ctx, "config", {}) or {}
        base = float(cfg.get("search_measure_scale", 1.0))
        n_fibers = int(cfg.get("n_fibers", 1) or 1)
        ifu_count = int(cfg.get("ifu_count", 1) or 1)
        exposure_scale = float(cfg.get("exposure_scale", 1.0) or 1.0)
        # sanitize
        n_fibers = max(n_fibers, 1)
        ifu_count = max(ifu_count, 1)
        exposure_scale = max(exposure_scale, 0.0)
        scale = base * n_fibers * ifu_count * exposure_scale
        if not np.isfinite(scale) or scale <= 0:
            return 1.0
        return float(scale)
    except Exception:
        return 1.0


def build_fake_lambda_pdf(wave_obs: np.ndarray, wave_min: float, wave_max: float, nbins: int = 200) -> Dict[str, np.ndarray | float]:
    """
    Build an empirical PDF over wavelength from virtual detections.

    Returns a cache dict with keys:
      - bins: bin edges (nbins+1)
      - pdf: probability density per Å, integrates to 1 over [wave_min, wave_max]
      - wave_min, wave_max, nbins
    """
    wave_min = float(wave_min)
    wave_max = float(wave_max)
    if not (np.isfinite(wave_min) and np.isfinite(wave_max)) or wave_max <= wave_min:
        raise ValueError("Invalid wave_min/wave_max for fake λ-PDF")
    nbins = int(max(nbins, 1))
    # Clean data into band
    x = np.asarray(wave_obs, dtype=float)
    x = x[np.isfinite(x)]
    x = x[(x >= wave_min) & (x <= wave_max)]
    if x.size == 0:
        # default to uniform density
        bins = np.linspace(wave_min, wave_max, nbins + 1)
        pdf = np.ones(nbins, dtype=float) / (wave_max - wave_min)
        return {"bins": bins, "pdf": pdf, "wave_min": wave_min, "wave_max": wave_max, "nbins": nbins}
    counts, bins = np.histogram(x, bins=nbins, range=(wave_min, wave_max))
    widths = np.diff(bins)
    area = float(np.sum(counts * widths))
    if area <= 0:
        pdf = np.ones(nbins, dtype=float) / (wave_max - wave_min)
    else:
        pdf = (counts / area).astype(float)
    return {"bins": bins, "pdf": pdf, "wave_min": wave_min, "wave_max": wave_max, "nbins": nbins}


def eval_fake_lambda_shape(lam: float, cache: Dict[str, np.ndarray | float]) -> float:
    """
    Evaluate the shape factor s(λ) with mean 1 over the band from the cached PDF.

    If pdf(λ) is the probability density (per Å) integrating to 1 over Δλ,
    define s(λ) = pdf(λ) * Δλ so that average s over the band is 1.

    Returns s(λ) >= 0. If λ outside band or cache invalid, returns 1.0.
    """
    try:
        bins = np.asarray(cache.get("bins"))
        pdf = np.asarray(cache.get("pdf"), dtype=float)
        wmin = float(cache.get("wave_min"))
        wmax = float(cache.get("wave_max"))
        if not (np.isfinite(lam) and np.isfinite(wmin) and np.isfinite(wmax)):
            return 1.0
        if lam < wmin or lam > wmax or bins.ndim != 1 or pdf.ndim != 1 or bins.size != pdf.size + 1:
            return 1.0
        # bin index
        idx = np.searchsorted(bins, lam, side="right") - 1
        idx = int(np.clip(idx, 0, pdf.size - 1))
        dlam = wmax - wmin
        if dlam <= 0:
            return 1.0
        s = float(pdf[idx] * dlam)
        if not np.isfinite(s) or s < 0:
            return 1.0
        return s
    except Exception:
        return 1.0



def save_fake_lambda_cache(path: str, cache: Dict[str, np.ndarray | float]) -> None:
    """
    Save a fake λ-PDF cache to disk (npz format).
    """
    try:
        import numpy as _np
        _np.savez_compressed(
            path,
            bins=_np.asarray(cache.get("bins")),
            pdf=_np.asarray(cache.get("pdf")),
            wave_min=float(cache.get("wave_min")),
            wave_max=float(cache.get("wave_max")),
            nbins=int(cache.get("nbins")),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save fake λ-PDF cache to {path}: {e}")


def load_fake_lambda_cache(path: str) -> Dict[str, np.ndarray | float]:
    """
    Load a fake λ-PDF cache saved by save_fake_lambda_cache().
    Returns a dict compatible with eval_fake_lambda_shape().
    """
    try:
        import numpy as _np
        with _np.load(path, allow_pickle=False) as data:
            bins = _np.asarray(data["bins"], dtype=float)
            pdf = _np.asarray(data["pdf"], dtype=float)
            wave_min = float(data["wave_min"]) if "wave_min" in data else float(bins.min())
            wave_max = float(data["wave_max"]) if "wave_max" in data else float(bins.max())
            nbins = int(data["nbins"]) if "nbins" in data else (bins.size - 1)
            return {"bins": bins, "pdf": pdf, "wave_min": wave_min, "wave_max": wave_max, "nbins": nbins}
    except Exception as e:
        raise RuntimeError(f"Failed to load fake λ-PDF cache from {path}: {e}")
