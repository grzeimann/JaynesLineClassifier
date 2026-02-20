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


def simulate_catalog_from_model(
    ctx,
    registry,
    ra_low: float,
    ra_high: float,
    dec_low: float,
    dec_high: float,
    wave_min: float,
    wave_max: float,
    flux_err: float = 1e-17,
    f_lim: Optional[float] = None,
    fake_rate_per_sr_per_A: float = 0.0,
    seed: Optional[int] = None,
    nz: int = 256,
    snr_min: Optional[float] = None,
) -> pd.DataFrame:
    """[DEPRECATED] Simulate a catalog via a Poisson point process using model expectations.

    This legacy simulator will be removed after the engine-aligned per-label simulation path is stable.
    Prefer using jlc.simulate.field.simulate_field(), which is the source of truth going forward.

    - Counts for LAE/OII are set by integrating the LF over flux and the comoving volume over the
      redshift interval implied by the wavelength band, multiplied by the sky solid angle.
    - Fakes are drawn from a uniform rate density per sr per Angstrom (fake_rate_per_sr_per_A).
    - Spatial positions are uniform within the sky box.
    - Measurements: flux_hat = F_true + N(0, flux_err), clipped to >=0.
    - No hard selection is applied post measurement; selection is encoded via S(F, lambda_obs) in the rate.
    """
    try:
        log("[jlc.simulate] WARNING: simulate_catalog_from_model() is LEGACY and will be removed; use simulate_field() instead.")
    except Exception:
        pass
    rng = np.random.default_rng(seed)

    # Solid angle
    omega = skybox_solid_angle_sr(ra_low, ra_high, dec_low, dec_high)
    omega = max(omega, 0.0)

    # Containers for results
    ras = []
    decs = []
    classes = []
    lam_obs_list = []
    F_true_list = []  # store true flux (before noise)
    F_hat_list = []   # store measured flux (after noise)
    F_err_list = []

    # Build config and flux grid
    cfg = PPPConfig(nz=nz, flux_grid=ctx.caches.get("flux_grid"))

    # Expectations summary for debugging/inspection
    exp_counts = {"total": 0.0}
    label_volumes = {}

    # Loop over labels in registry
    ppp_debug = False
    try:
        ppp_debug = bool(getattr(ctx, "config", {}).get("ppp_debug", False))
    except Exception:
        ppp_debug = False
    debug_info = []
    for label_name in registry.labels:
        model = registry.model(label_name)
        # In virtual volume mode, suppress physical labels entirely
        try:
            if str(getattr(ctx, "config", {}).get("volume_mode", "real")).lower() == "virtual" and label_name != "fake":
                # still record zeros for expectations
                exp_counts[label_name] = 0.0
                if ppp_debug:
                    debug_info.append({"label": label_name, "note": "virtual_mode_suppressed"})
                continue
        except Exception:
            pass
        if label_name == "fake": 
            # Fakes: homogeneous PPP in (sky, lambda)
            rho = max(float(fake_rate_per_sr_per_A), 0.0)
            mu = rho * omega * max(0.0, float(wave_max - wave_min))
            exp_counts["fake"] = float(mu)
            exp_counts["total"] += float(mu)
            if ppp_debug:
                try:
                    log(f"[jlc.ppp.debug] label=fake Ω={omega:.3e} sr, Δλ={wave_max - wave_min:.1f} Å, ρ={rho:.3e} → μ≈{mu:.3e}")
                except Exception:
                    pass
                debug_info.append({"label": "fake", "omega_sr": omega, "band_A": float(wave_max - wave_min), "rho": rho, "mu": mu})
            n_fake = _poisson_safe(rng, mu)
            n_fake = _cap_events("fake", n_fake, mu)
            if n_fake > 0:
                # RA uniform; DEC sampled to be uniform per steradian via sin(dec) transform
                ra = rng.uniform(ra_low, ra_high, size=n_fake)
                s1 = np.sin(np.deg2rad(dec_low)); s2 = np.sin(np.deg2rad(dec_high))
                u = rng.uniform(min(s1, s2), max(s1, s2), size=n_fake)
                dec = np.rad2deg(np.arcsin(u))
                lam = rng.uniform(wave_min, wave_max, size=n_fake)
                # Flux prior for fakes: log-normal around f_lim (mostly below)
                if f_lim is None:
                    base = 1e-17
                else:
                    base = max(f_lim, 1e-20)
                mu_ln = np.log(base) - 2.0
                sigma_ln = 1.0
                F_true = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_fake)
                F_err = np.full(n_fake, float(flux_err))
                F_hat = np.clip(F_true + rng.normal(0.0, F_err), 0.0, None)

                # Apply selection completeness to fakes as a detection probability at (F_true, lambda)
                # completeness(F, λ) supports vector F and scalar λ; loop over λ to build per-event C
                try:
                    sel = ctx.selection
                    C = np.empty(n_fake, dtype=float)
                    for i in range(n_fake):
                        Ci = sel.completeness(np.asarray([F_true[i]], dtype=float), float(lam[i]))
                        C[i] = float(Ci[0]) if np.size(Ci) > 0 else 0.0
                    C = np.clip(C, 0.0, 1.0)
                    keep = rng.uniform(0.0, 1.0, size=n_fake) < C
                except Exception:
                    # If selection evaluation fails, keep all to avoid crashing
                    keep = np.ones(n_fake, dtype=bool)

                if np.any(keep):
                    ra = ra[keep]
                    dec = dec[keep]
                    lam = lam[keep]
                    F_true = F_true[keep]
                    F_err = F_err[keep]
                    F_hat = F_hat[keep]

                    ras.append(ra)
                    decs.append(dec)
                    classes.append(np.array(["fake"] * ra.size))
                    lam_obs_list.append(lam)
                    F_true_list.append(F_true)
                    F_hat_list.append(F_hat)
                    F_err_list.append(F_err)
            continue

        # Line-emitter labels (e.g., LAE, OII)
        # Identify rest wavelength
        rest_wave = float(getattr(model, "rest_wave", np.nan))
        if not np.isfinite(rest_wave) or rest_wave <= 0:
            continue

        z_grid, F_grid, zmin, zmax = _compute_label_grids(ctx, rest_wave, wave_min, wave_max, cfg)
        if z_grid is None:
            continue

        # Rate per sr per z per F
        rate, dVdz, dL2 = _rate_grid_per_sr(ctx, getattr(model, "lf"), ctx.selection, rest_wave, z_grid, F_grid)

        # Marginal over F to get rate over z
        r_z = integrate_rate_over_flux(rate, F_grid)  # shape (nz,)

        # Expected count over sky Ω: μ = Ω * ∫ r_z dz
        mu_label = expected_count_from_rate(r_z, z_grid, omega)
        exp_counts[label_name] = float(mu_label)
        exp_counts["total"] += float(mu_label)

        # Comoving volume probed for this label (Ω × ∫ dV/dz dz)
        V_label = expected_volume_from_dVdz(dVdz, z_grid, omega)
        label_volumes[label_name] = V_label

        if ppp_debug:
            # Estimate fraction of LF mass within bounds over the explored L range
            try:
                lf = getattr(model, "lf")
                L_grid_sample = 4.0 * np.pi * dL2[int(len(dL2)//2)] * F_grid  # mid-z slice
                phi_all = lf.phi(L_grid_sample)
                mass_all = float(np.trapz(phi_all, x=L_grid_sample))
                if getattr(lf, "Lmin", None) is not None or getattr(lf, "Lmax", None) is not None:
                    mask = np.ones_like(L_grid_sample, dtype=bool)
                    if getattr(lf, "Lmin", None) is not None:
                        mask &= (L_grid_sample >= float(lf.Lmin))
                    if getattr(lf, "Lmax", None) is not None:
                        mask &= (L_grid_sample <= float(lf.Lmax))
                    phi_masked = np.where(mask, phi_all, 0.0)
                    mass_in = float(np.trapz(phi_masked, x=L_grid_sample))
                    frac_in = (mass_in / mass_all) if (mass_all > 0) else np.nan
                else:
                    frac_in = 1.0
            except Exception:
                frac_in = np.nan
            # Average selection over F and z (rough summary)
            try:
                lam_obs = rest_wave * (1.0 + z_grid)
                S_avg = []
                for i in range(len(z_grid)):
                    S = ctx.selection.completeness(F_grid, float(lam_obs[i]))
                    # weight by phi_F at this z for relevance
                    dLdF_i = 4.0 * np.pi * dL2[i]
                    Lg = 4.0 * np.pi * dL2[i] * F_grid
                    phiF = getattr(model, "lf").phi(Lg) * dLdF_i
                    w = phiF
                    w_sum = np.sum(w)
                    S_avg.append(float(np.sum(S * (w / w_sum))) if (np.isfinite(w_sum) and w_sum > 0) else float(np.mean(S)))
                S_mean = float(np.mean(S_avg)) if len(S_avg) > 0 else float("nan")
            except Exception:
                S_mean = float("nan")
            # Also compute the expected count without selection (diagnostic only)
            try:
                class _UnitSelection:
                    def completeness(self, F, wave):
                        return np.ones_like(F, dtype=float)
                rate_nosel, _dVdz0, _dL20 = _rate_grid_per_sr(ctx, getattr(model, "lf"), _UnitSelection(), rest_wave, z_grid, F_grid)
                r_z0 = integrate_rate_over_flux(rate_nosel, F_grid)
                mu_nosel = expected_count_from_rate(r_z0, z_grid, omega)
                S_eff = (mu_label / mu_nosel) if (np.isfinite(mu_nosel) and mu_nosel > 0) else np.nan
            except Exception:
                mu_nosel = float('nan')
                S_eff = float('nan')
            try:
                log(f"[jlc.ppp.debug] label={label_name} Ω={omega:.3e} sr, λ_rest={rest_wave:.2f} Å, z∈[{zmin:.3f},{zmax:.3f}], V≈{V_label:.3e} Mpc^3, μ≈{mu_label:.3e}, μ0(no S)≈{mu_nosel:.3e}, S_eff≈{S_eff:.3f}, frac_LF_in_bounds≈{frac_in:.3f}, ⟨S⟩≈{S_mean:.3f}")
            except Exception:
                pass
            debug_info.append({
                "label": label_name,
                "omega_sr": omega,
                "rest_wave": rest_wave,
                "zmin": float(zmin),
                "zmax": float(zmax),
                "V_Mpc3": V_label,
                "mu": mu_label,
                "mu_no_selection": mu_nosel,
                "S_eff": S_eff,
                "frac_LF_in_bounds": frac_in,
                "S_mean": S_mean,
            })

        n_label = _poisson_safe(rng, max(mu_label, 0.0))
        n_label = _cap_events(label_name, n_label, mu_label)
        if n_label <= 0:
            continue

        # Sample redshifts proportionally to r(z) Δz
        idx_z, p_z = sample_z_indices(rng, z_grid, r_z, n_label)
        if idx_z.size == 0:
            continue
        z_samp = z_grid[idx_z]
        lam_obs = rest_wave * (1.0 + z_samp)

        # For each sampled z, sample F from conditional p(F|z) ∝ rate(z,F)
        F_samp = np.empty(n_label, dtype=float)
        for k in range(n_label):
            i = idx_z[k]
            F_samp[k] = sample_F_given_z(rng, rate[i, :], F_grid)

        # Observational noise
        F_err = np.full(n_label, float(flux_err))
        F_hat = np.clip(F_samp + rng.normal(0.0, F_err), 0.0, None)

        # Sky positions: RA uniform; DEC sampled to be uniform per steradian via sin(dec) transform
        ra = rng.uniform(ra_low, ra_high, size=n_label)
        s1 = np.sin(np.deg2rad(dec_low)); s2 = np.sin(np.deg2rad(dec_high))
        u = rng.uniform(min(s1, s2), max(s1, s2), size=n_label)
        dec = np.rad2deg(np.arcsin(u))

        # Append
        ras.append(ra)
        decs.append(dec)
        classes.append(np.array([label_name] * n_label))
        lam_obs_list.append(lam_obs)
        F_true_list.append(F_samp)
        F_hat_list.append(F_hat)
        F_err_list.append(F_err)

    # Concatenate all
    if len(classes) == 0:
        # Expose expected counts and volumes in context for downstream use
        try:
            if hasattr(ctx, "config") and isinstance(ctx.config, dict):
                ctx.config["ppp_expected_counts"] = dict(exp_counts)
                ctx.config["ppp_label_volumes"] = dict(label_volumes)
                # Optionally attach detailed PPP debug info for downstream diagnostics
                if bool(getattr(ctx, "config", {}).get("ppp_debug", False)):
                    ctx.config["ppp_debug_info"] = list(debug_info)
        except Exception:
            pass
        # Print expectations even if no objects realized
        try:
            lae_mu = exp_counts.get("lae", 0.0)
            oii_mu = exp_counts.get("oii", 0.0)
            fake_mu = exp_counts.get("fake", 0.0)
            lae_V = label_volumes.get("lae", float("nan"))
            oii_V = label_volumes.get("oii", float("nan"))
            total_mu = exp_counts.get("total", 0.0)
            log(f"[jlc.simulate] Expected counts (incl. selection for LAE/OII; fakes from ρ over band): lae≈{lae_mu:.3e}, oii≈{oii_mu:.3e}, fake≈{fake_mu:.3e}; total≈{total_mu:.3e}")
            log(f"[jlc.simulate] Volumes (Mpc^3): lae≈{lae_V:.3e}, oii≈{oii_V:.3e}")
        except Exception:
            pass
        return pd.DataFrame(columns=["ra", "dec", "true_class", "wave_obs", "flux_true", "flux_hat", "flux_err"]).copy()

    ra_all = np.concatenate(ras)
    dec_all = np.concatenate(decs)
    cls_all = np.concatenate(classes)
    lam_all = np.concatenate(lam_obs_list)
    Ftrue_all = np.concatenate(F_true_list)
    Fhat_all = np.concatenate(F_hat_list)
    Ferr_all = np.concatenate(F_err_list)

    df = pd.DataFrame({
        "ra": ra_all,
        "dec": dec_all,
        "true_class": cls_all,
        "wave_obs": lam_all,
        "flux_true": Ftrue_all,
        "flux_hat": Fhat_all,
        "flux_err": Ferr_all,
    })

    # Compute S/N and apply S/N filter if requested
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(df["flux_err"].values > 0, df["flux_hat"].values / df["flux_err"].values, 0.0)
    df["snr"] = snr
    if snr_min is not None and np.isfinite(snr_min) and snr_min > 0:
        before = len(df)
        df = df.loc[df["snr"] >= float(snr_min)].reset_index(drop=True)
        after = len(df)
        try:
            log(f"[jlc.simulate] Applied S/N cut: snr_min={snr_min:.2f}. Kept {after}/{before} rows ({(after/max(before,1))*100:.1f}%).")
        except Exception:
            pass

    # Expose expected counts and volumes in context for downstream use
    try:
        if hasattr(ctx, "config") and isinstance(ctx.config, dict):
            ctx.config["ppp_expected_counts"] = dict(exp_counts)
            ctx.config["ppp_label_volumes"] = dict(label_volumes)
            # Optionally attach detailed PPP debug info for downstream diagnostics
            if bool(getattr(ctx, "config", {}).get("ppp_debug", False)):
                ctx.config["ppp_debug_info"] = list(debug_info)
    except Exception:
        pass

    # Print expectations summary
    try:
        lae_mu = exp_counts.get("lae", 0.0)
        oii_mu = exp_counts.get("oii", 0.0)
        fake_mu = exp_counts.get("fake", 0.0)
        lae_V = label_volumes.get("lae", float("nan"))
        oii_V = label_volumes.get("oii", float("nan"))
        total_mu = exp_counts.get("total", 0.0)
        log(f"[jlc.simulate] Expected counts (incl. selection for LAE/OII; fakes from ρ over band): lae≈{lae_mu:.3e}, oii≈{oii_mu:.3e}, fake≈{fake_mu:.3e}; total≈{total_mu:.3e}")
        log(f"[jlc.simulate] Volumes (Mpc^3): lae≈{lae_V:.3e}, oii≈{oii_V:.3e}")
    except Exception:
        pass

    # Shuffle rows to mix labels
    if len(df) > 1:
        df = df.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1)).reset_index(drop=True)
    return df
