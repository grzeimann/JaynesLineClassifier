from __future__ import annotations
"""
Shared LF + PPP helper functions used by physical labels (LAE, OII, ...).

This module centralizes geometry, unit conversions, and observed-space rate
integrands so we can reason about and test the math in a single place.

Notation and units:
- Cosmology methods (luminosity_distance, dV_dz) are assumed to return distances
  in Mpc and comoving volume element in Mpc^3 / sr / dz, respectively.
- Flux F is in CGS (erg s^-1 cm^-2) as usual.
- Luminosity L is in CGS (erg s^-1).
- The observed-space rate density returned by helpers integrates to counts per
  steradian per Angstrom across flux: r(\lambda) = \int dF r_F(F,\lambda).
"""
import numpy as np
from typing import Tuple

MPC_TO_CM = 3.0856775814913673e24  # cm per Mpc
TWOPI = 2.0 * np.pi
FOURPI = 4.0 * np.pi


# ----------------------- Conversions / Geometry -----------------------

def redshift_from_lambda(lam_obs: float, rest_wave: float) -> float:
    """Compute redshift z from observed wavelength and rest wavelength.

    Parameters
    ----------
    lam_obs : float
        Observed wavelength (Angstrom).
    rest_wave : float
        Rest-frame line wavelength (Angstrom).

    Returns
    -------
    float
        z = lam_obs / rest_wave - 1, with guards for invalid inputs (<=0 -> nan).
    """
    try:
        lam = float(lam_obs)
        rw = float(rest_wave)
        if not (np.isfinite(lam) and np.isfinite(rw)) or lam <= 0 or rw <= 0:
            return float("nan")
        return float(lam / rw - 1.0)
    except Exception:
        return float("nan")


def luminosity_from_flux(F: np.ndarray | float, z: float, cosmo) -> np.ndarray:
    """Convert observed flux F to luminosity L at redshift z using cosmology.

    L = 4π d_L(z)^2 F, with d_L in cm (converted from Mpc).
    """
    try:
        dL_mpc = float(cosmo.luminosity_distance(float(z)))
    except Exception:
        dL_mpc = float("nan")
    dL_cm = dL_mpc * MPC_TO_CM
    F = np.asarray(F, dtype=float)
    return FOURPI * (dL_cm ** 2) * F


def dL2_cm_from_z(z: float, cosmo) -> float:
    """Return d_L(z)^2 in cm^2 (helper to avoid redundant conversions)."""
    try:
        dL_mpc = float(cosmo.luminosity_distance(float(z)))
        return float((dL_mpc * MPC_TO_CM) ** 2)
    except Exception:
        return float("nan")


def jac_dz_dlambda(rest_wave: float) -> float:
    """Return |dz/dλ| for λ_obs = rest_wave * (1+z) → dz/dλ = 1/rest_wave."""
    try:
        rw = float(rest_wave)
        return 1.0 / rw if (np.isfinite(rw) and rw > 0) else float("nan")
    except Exception:
        return float("nan")


# ----------------------- LF / Selection utilities -----------------------

def phi_L_from_lf(lf, L: np.ndarray) -> np.ndarray:
    """Evaluate LF ϕ(L) using provided LF object (e.g., SchechterLF)."""
    try:
        return np.asarray(lf.phi(L), dtype=float)
    except Exception:
        return np.zeros_like(L, dtype=float)


def phi_flux_from_lf(lf, F: np.ndarray, z: float, cosmo) -> np.ndarray:
    """Map LF to flux-space at redshift z via L(F,z) and return ϕ(L(F,z))."""
    L = luminosity_from_flux(F, z, cosmo)
    return phi_L_from_lf(lf, L)


def completeness(selection, F: np.ndarray, lam_obs: float, label_name: str | None = None) -> np.ndarray:
    """S/N-based completeness wrapper using SelectionModel.completeness_sn_array.

    Falls back to ones if selection/noise/SN model are not configured.
    """
    F = np.asarray(F, dtype=float)
    if selection is None:
        return np.ones_like(F, dtype=float)
    try:
        lname = str(label_name) if label_name is not None else "all"
        C = selection.completeness_sn_array(lname, F, float(lam_obs))
        C = np.clip(np.asarray(C, dtype=float), 0.0, 1.0)
        return C
    except Exception:
        return np.ones_like(F, dtype=float)


# ----------------------- Rate density (observed space) -----------------------

def rate_density_integrand_per_flux(
    lf,
    selection,
    F_grid: np.ndarray,
    lam_obs: float,
    rest_wave: float,
    z: float,
    cosmo,
    *,
    label_name: str | None = None,
) -> Tuple[np.ndarray, float, float]:
    """Return r_F(F,λ) array per (sr·Å·flux) and auxiliary factors.

    r_F(F,λ) = dV/dz(z) · [ϕ(L(F,z)) · dL/dF] · S(F,λ) · |dz/dλ|.

    Returns
    -------
    (rF, dVdz, dL2_cm)
      rF : np.ndarray over F_grid with dimensions per (sr·Å·flux)
      dVdz : scalar comoving volume element Mpc^3 / sr / dz (for diagnostics)
      dL2_cm : scalar d_L(z)^2 in cm^2 (for diagnostics)
    """
    F = np.asarray(F_grid, dtype=float)
    # Cosmology factors
    try:
        dVdz = float(cosmo.dV_dz(float(z)))
    except Exception:
        dVdz = 0.0
    dL2 = dL2_cm_from_z(z, cosmo)
    dLdF = FOURPI * dL2  # derivative of L wrt F at fixed z
    # LF and selection
    phi = phi_flux_from_lf(lf, F, z, cosmo)
    S = completeness(selection, F, lam_obs, label_name=label_name)
    # Jacobian |dz/dλ|
    jac = jac_dz_dlambda(rest_wave)
    rF = dVdz * (phi * dLdF) * S * jac
    # sanitize
    rF = np.where(np.isfinite(rF) & (rF >= 0.0), rF, 0.0)
    return rF, dVdz, dL2


def rate_density_local(
    row,
    ctx,
    rest_wave: float,
    lf,
    selection,
    *,
    label_name: str | None = None,
) -> float:
    """Observed-space rate density at measured flux: r(λ, F_hat) per sr per Å.

    New behavior: do NOT integrate over the shared FluxGrid. Instead, given the
    measured flux_hat and its uncertainty, return

        r(λ, F_hat) = ∫ dF_true r_F(F_true, λ) · p(F_hat | F_true)

    where r_F is the per-(sr·Å·flux) integrand from the LF and selection, and
    p(F_hat|F_true) is the flux measurement model (Gaussian with sigma=flux_err).

    Multiplies by effective_search_measure(row, ctx) if available. Respects
    ctx.config["volume_mode"] == "virtual" by returning 0 for physical labels.
    """
    # Respect virtual mode for physical labels (caller should decide applicability)
    try:
        if str(getattr(ctx, "config", {}).get("volume_mode", "real")).lower() == "virtual":
            return 0.0
    except Exception:
        pass
    # Wavelength and redshift
    try:
        lam = float(row.get("wave_obs"))
    except Exception:
        lam = float("nan")
    if not (np.isfinite(lam) and lam > 0):
        return 0.0
    z = redshift_from_lambda(lam, rest_wave)
    if not (np.isfinite(z) and z > 0):
        return 0.0
    # Measured flux and error
    try:
        F_hat = float(row.get("flux_hat", np.nan))
        sigma = float(row.get("flux_err", np.nan))
    except Exception:
        F_hat, sigma = float("nan"), float("nan")
    if not (np.isfinite(F_hat) and F_hat >= 0):
        return 0.0
    # If error is invalid or zero, evaluate at delta: r_F(F_hat, λ)
    if not (np.isfinite(sigma) and sigma > 0):
        F_eval = np.array([F_hat], dtype=float)
        rF, _dv, _dl2 = rate_density_integrand_per_flux(lf, selection, F_eval, lam, rest_wave, z, ctx.cosmo, label_name=label_name)
        r = float(rF[0])
        try:
            from jlc.rates.observed_space import effective_search_measure
            r *= float(effective_search_measure(row, ctx))
        except Exception:
            pass
        return max(r, 0.0)

    fg = getattr(getattr(ctx, "caches", {}), "get", lambda k, d=None: None)("flux_grid") if hasattr(getattr(ctx, "caches", {}), "get") else getattr(ctx.caches, "flux_grid", None)
    if fg is None:
        fg = getattr(ctx.caches if hasattr(ctx, "caches") else {}, "flux_grid", None)
    if fg is not None and hasattr(fg, "grid"):
        Fg, _logw = fg.grid(row)
        Fg = np.asarray(Fg, dtype=float)
        # sanitize: keep finite and non-negative
        Fg = Fg[np.isfinite(Fg) & (Fg >= 0.0)]
        if Fg.size >= 2:
            F_grid_true = Fg


    # Evaluate LF×selection integrand r_F(F_true, λ)
    rF, _dVdz, _dL2 = rate_density_integrand_per_flux(lf, selection, F_grid_true, lam, rest_wave, z, ctx.cosmo, label_name=label_name)
    # Measurement PDF p(F_hat | F_true) assuming Gaussian noise on flux_hat
    # p = N(F_hat; mean=F_true, sigma)
    inv_s = 1.0 / sigma
    norm = inv_s / np.sqrt(2.0 * np.pi)
    resid = (F_hat - F_grid_true) * inv_s
    p_meas = norm * np.exp(-0.5 * resid * resid)
    # Convolution integral over F_true
    integrand = rF * p_meas
    r = float(np.trapz(integrand, x=F_grid_true))
    return max(r, 0.0)

