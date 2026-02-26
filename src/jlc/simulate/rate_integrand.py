from __future__ import annotations
import numpy as np
from typing import Any

from jlc.population.schechter import SchechterLF
from jlc.cosmology.lookup import AstropyCosmology
from .diagnostics import get_completeness_tracer


def _rest_wavelength_for_label(name: str) -> float:
    """Return rest wavelength (Å) for known labels; defaults to LAE if unknown.
    Values align with defaults in configs/priors.
    """
    n = str(name).strip().lower()
    if n in ("lae", "lya", "lyalpha", "ly-α", "lyalpha_emitter"):
        return 1215.67
    if n in ("oii", "[oii]", "oii_3727", "o2"):
        return 3727.0
    # Fallback to LAE rest λ to keep rate finite; can be overridden by passing a
    # SchechterLF already in flux space (then mapping below is harmless since
    # typical use for 'fake' bypasses LF or uses completeness ~ 1).
    return 1215.67


def rate_density_integrand_per_flux(
    F_true_grid: np.ndarray,
    lam: float,
    label: str,
    lf_params: Any,
    completeness_provider: Any,
) -> np.ndarray:
    """Compute differential rate density per (sr·Å·flux) at a given λ and label.

    Implements the physical mapping from flux to luminosity for Schechter LFs:
      - z = lam / lam_rest(label) − 1
      - L = 4π d_L(z)^2 F
      - r_F(F, λ) = (dV/dz)(z) · φ(L) · (dL/dF) · C(F, λ) · |dz/dλ|
        with dL/dF = 4π d_L^2 and |dz/dλ| = 1 / lam_rest.

    Parameters
    ----------
    F_true_grid : array-like (flux)
        Flux grid (erg s^-1 cm^-2) to evaluate over.
    lam : float
        Observed wavelength (Å) for evaluating completeness and cosmology mapping.
    label : str
        Label name for completeness lookup and rest-wavelength mapping.
    lf_params : SchechterLF | Any
        Either a SchechterLF instance or a mapping sufficient to build one.
    completeness_provider : CompletenessProvider
        Object exposing completeness(F_true_grid, lam, label) -> array in [0,1].

    Returns
    -------
    r_F : np.ndarray
        Differential rate density per (sr·Å·flux) evaluated on F_true_grid.
    """
    F = np.asarray(F_true_grid, dtype=float)
    if F.size == 0:
        return np.asarray(F, dtype=float)

    # Cosmology and redshift mapping
    lam_rest = float(_rest_wavelength_for_label(label))
    z = float(max(lam / max(lam_rest, 1e-12) - 1.0, 1e-8))
    cosmo = AstropyCosmology()
    # Luminosity distance in cm
    dL_Mpc = float(cosmo.luminosity_distance(z))
    dL_cm = dL_Mpc * 3.0856775814913673e24  # exact Mpc to cm
    # Jacobians
    dL_dF = 4.0 * np.pi * (dL_cm ** 2)
    abs_dz_dlam = 1.0 / max(lam_rest, 1e-12)
    dV_dz_sr = float(cosmo.dV_dz(z))  # Mpc^3 / sr

    # Obtain an LF model instance
    if isinstance(lf_params, SchechterLF):
        lf = lf_params
    elif isinstance(lf_params, dict):
        lf = SchechterLF(**lf_params)
    else:
        lf = lf_params

    # Evaluate φ(L) at mapped luminosities
    try:
        L = dL_dF * F  # erg s^-1
        phi_L = np.asarray(lf.phi(L), dtype=float)
    except Exception:
        phi_L = np.zeros_like(F, dtype=float)

    # Completeness in [0,1]
    tracer = get_completeness_tracer()
    try:
        C = np.asarray(completeness_provider.completeness(F, float(lam), str(label)), dtype=float)
    except Exception:
        # Record site of failure and fall back conservatively to ones here (provider-level guards handle zeros on bad noise)
        try:
            tracer.on_exception("integrand", str(label))
        except Exception:
            pass
        C = np.ones_like(F, dtype=float)

    # Clamp to sane ranges
    phi_L = np.where(np.isfinite(phi_L) & (phi_L >= 0), phi_L, 0.0)
    C = np.clip(np.where(np.isfinite(C), C, 0.0), 0.0, 1.0)

    # Observe for diagnostics (no effect on math)
    try:
        tracer.observe(F, C, str(label), float(lam))
    except Exception:
        pass

    # Differential rate density per (sr·Å·flux)
    # Units: [Mpc^3/sr]·[1/(Mpc^3·L)]·[L/F]·[1]·[1/Å] → [1/(sr·Å·F)]
    r_F = (dV_dz_sr * phi_L * dL_dF * C * abs_dz_dlam)
    return np.asarray(r_F, dtype=float)