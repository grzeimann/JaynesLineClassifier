import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from jlc.engine.flux_grid import FluxGrid
from jlc.utils.logging import log

# Hard safety cap to prevent infeasible allocations when sampling extremely large
# Poisson counts. If expected counts exceed this, we cap and warn. This avoids
# numpy overflows and out-of-memory errors during rng.choice and array creation.
MAX_EVENTS_PER_LABEL = 5_000_000

# Unit conversion: luminosity distance is returned in Mpc by the cosmology,
# but luminosities in the Schechter LF are in CGS (erg/s). Convert Mpc->cm
# before forming L = 4π d_L^2 F.
MPC_TO_CM = 3.085677581491367e24


@dataclass
class PPPConfig:
    # Discretization controls
    nz: int = 256
    # Use the same flux grid as engine by default; can override by supplying FluxGrid
    flux_grid: Optional[FluxGrid] = None
    # Fake background rate density per steradian per Angstrom
    fake_rate_per_sr_per_A: float = 0.0


def _poisson_safe(rng: np.random.Generator, lam: float) -> int:
    """Draw from Poisson(lam) with guards for extremely large lam.

    numpy's Generator.poisson raises ValueError for excessively large lam.
    For lam >= 1e6 we switch to a normal approximation N(lam, lam) and
    return a non-negative integer. For non-finite or non-positive lam, return 0.
    """
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0.0:
        return 0
    if lam < 1e6:
        return int(rng.poisson(lam))
    # Normal approximation for large mean
    n = rng.normal(lam, np.sqrt(lam))
    return int(max(0, np.floor(n + 0.5)))


def _cap_events(label: str, n: int, mu: float) -> int:
    """Clamp the number of simulated events to a safe cap to avoid overflow/OOM."""
    n_int = int(max(0, n))
    if n_int > MAX_EVENTS_PER_LABEL:
        try:
            log(f"[jlc.simulate] Capping {label} events: drew n={n_int} from mu≈{mu:.3e}, "
                f"limiting to {MAX_EVENTS_PER_LABEL} to avoid overflow/memory issues.")
        except Exception:
            # Be robust if stdout is not available
            pass
        return int(MAX_EVENTS_PER_LABEL)
    return n_int


def skybox_solid_angle_sr(ra_low: float, ra_high: float, dec_low: float, dec_high: float) -> float:
    """Approximate solid angle (steradians) of a rectangular RA/Dec box.

    RA/Dec are in degrees. Assumes small box, no RA wrap-around handling beyond simple diff.
    Ω = ΔRA_rad * (sin(dec_high) - sin(dec_low)).
    """
    # Normalize ranges
    ra1, ra2 = float(ra_low), float(ra_high)
    dec1, dec2 = float(dec_low), float(dec_high)
    dra_deg = max(0.0, ra2 - ra1)
    dec1_rad = np.deg2rad(dec1)
    dec2_rad = np.deg2rad(dec2)
    dra_rad = np.deg2rad(dra_deg)
    omega = abs(dra_rad) * abs(np.sin(dec2_rad) - np.sin(dec1_rad))
    return float(max(omega, 0.0))


def _compute_label_grids(ctx, rest_wave: float, wave_min: float, wave_max: float, cfg: PPPConfig):
    # Redshift limits from wavelength band
    zmin = max(wave_min / rest_wave - 1.0, 1e-8)
    zmax = max(wave_max / rest_wave - 1.0, 0.0)
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return None, None, None, None
    z_grid = np.linspace(zmin, zmax, int(max(cfg.nz, 2)))
    # Flux grid from FluxGrid cache or provided
    fg: FluxGrid = cfg.flux_grid or ctx.caches.get("flux_grid") or FluxGrid()
    F_grid, _log_w = fg.grid(pd.Series({}))
    return z_grid, F_grid, zmin, zmax


def _rate_grid_per_sr(ctx, lf, selection, rest_wave: float, z_grid: np.ndarray, F_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute r(z, F) per sr per unit z per unit F for a line label.

    r(z,F) = dV/dz(z) * phi_F(F|z) * S(F, lambda_obs), where phi_F = phi_L(L) * dL/dF, L=4π d_L(z)^2 F.
    Returns (rate_grid, dVdz_grid, dL2_grid) for possible reuse.
    """
    # Precompute cosmology terms on z grid
    dL = ctx.cosmo.luminosity_distance(z_grid)  # Mpc
    # Convert to cm for luminosity computation (LF expects L in erg/s)
    dL2 = (dL * MPC_TO_CM) ** 2  # cm^2
    dVdz = ctx.cosmo.dV_dz(z_grid)  # Mpc^3 / sr / dz

    # Observed wavelength at each z
    lam_obs = rest_wave * (1.0 + z_grid)

    # Build rate grid
    nz = z_grid.size
    nF = F_grid.size
    rate = np.zeros((nz, nF), dtype=float)

    # Precompute selection matrix S(F, lambda)
    # selection.completeness expects array F and single wavelength; we'll loop over z
    for i in range(nz):
        L_grid = 4.0 * np.pi * dL2[i] * F_grid
        phi_L = lf.phi(L_grid)  # number density per luminosity
        # Backstop: if LF exposes luminosity bounds, zero out contributions outside [Lmin, Lmax]
        Lmin = getattr(lf, "Lmin", None)
        Lmax = getattr(lf, "Lmax", None)
        if (Lmin is not None) or (Lmax is not None):
            mask = np.ones_like(L_grid, dtype=bool)
            if Lmin is not None:
                mask &= (L_grid >= float(Lmin))
            if Lmax is not None:
                mask &= (L_grid <= float(Lmax))
            # Ensure mask applied even if lf.phi didn't enforce bounds internally
            phi_L = np.where(mask, phi_L, 0.0)
        dLdF = 4.0 * np.pi * dL2[i]
        phi_F = phi_L * dLdF
        S = selection.completeness(F_grid, float(lam_obs[i]))
        # rate per z per F per sr
        rate[i, :] = dVdz[i] * phi_F * S

    return rate, dVdz, dL2


def _sample_from_weights(rng: np.random.Generator, weights: np.ndarray) -> int:
    w = np.asarray(weights, dtype=float)
    total = np.sum(w)
    if not np.isfinite(total) or total <= 0:
        return -1
    p = w / total
    # Use integer choice index
    return int(rng.choice(len(w), p=p))


# ==========================
# Modular helper functions
# ==========================

def integrate_rate_over_flux(rate: np.ndarray, F_grid: np.ndarray) -> np.ndarray:
    """Integrate rate(z, F) over F to produce r(z) on the same z-grid.

    Uses trapezoidal rule along the F axis.
    """
    return np.trapz(rate, x=F_grid, axis=1)


essentially_zero = lambda x: (not np.isfinite(x)) or (x <= 0.0)


def expected_count_from_rate(r_z: np.ndarray, z_grid: np.ndarray, omega_sr: float) -> float:
    """μ = Ω * ∫ r(z) dz"""
    if r_z.size == 0:
        return 0.0
    mu = float(omega_sr * np.trapz(r_z, x=z_grid))
    if not np.isfinite(mu):
        mu = 0.0
    return mu


def expected_volume_from_dVdz(dVdz: np.ndarray, z_grid: np.ndarray, omega_sr: float) -> float:
    """Comoving volume probed for a label: V = Ω * ∫ dV/dz dz"""
    V = float(omega_sr * np.trapz(dVdz, x=z_grid))
    if not np.isfinite(V):
        V = 0.0
    return V


def sample_z_indices(rng: np.random.Generator, z_grid: np.ndarray, r_z: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample z-bin indices proportionally to r(z) Δz. Returns (idx, p_z)."""
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    dz = np.gradient(z_grid)
    w_z = r_z * dz
    s = np.sum(w_z)
    if (not np.isfinite(s)) or (s <= 0):
        return np.array([], dtype=int), w_z
    p_z = w_z / s
    idx_z = rng.choice(z_grid.size, size=n, p=p_z)
    return idx_z.astype(int), p_z


def sample_F_given_z(rng: np.random.Generator, rate_row: np.ndarray, F_grid: np.ndarray) -> float:
    """Sample F conditional on a fixed z, using rate(z, F) as weights (with ΔF)."""
    dF = np.gradient(F_grid)
    wF = rate_row * dF
    s = np.sum(wF)
    if (not np.isfinite(s)) or (s <= 0):
        return float(F_grid[int(rng.integers(0, F_grid.size))])
    pF = wF / s
    idxF = int(rng.choice(F_grid.size, p=pF))
    return float(F_grid[idxF])


