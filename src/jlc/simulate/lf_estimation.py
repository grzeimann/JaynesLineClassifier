import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

# Cosmology returns d_L in Mpc; convert to cm for L = 4π d_L^2 F in CGS.
MPC_TO_CM = 3.085677581491367e24

LABEL_REST_WAVE = {
    "lae": 1215.67,
    "oii": 3727.0,
}

def build_zgrid_and_dVdz(cosmo, rest_wave: float, wave_min: float, wave_max: float, nz: int = 512):
    """Build a redshift grid within [wave_min, wave_max] for a given rest wavelength and dV/dz on that grid.

    Returns (z_grid, dVdz, lam_obs_grid).
    """
    if not np.isfinite(rest_wave) or rest_wave <= 0:
        return np.array([]), np.array([]), np.array([])
    zmin = max(float(wave_min) / rest_wave - 1.0, 1e-8)
    zmax = max(float(wave_max) / rest_wave - 1.0, 0.0)
    if not (np.isfinite(zmin) and np.isfinite(zmax)) or zmax <= zmin:
        return np.array([]), np.array([]), np.array([])
    z_grid = np.linspace(zmin, zmax, int(max(nz, 4)))
    try:
        dVdz = np.asarray(cosmo.dV_dz(z_grid), dtype=float)
    except Exception:
        dVdz = np.array([cosmo.dV_dz(float(z)) for z in z_grid], dtype=float)
    lam_obs = rest_wave * (1.0 + z_grid)
    return z_grid, dVdz, lam_obs

def skybox_solid_angle_sr(ra_low: float, ra_high: float, dec_low: float, dec_high: float) -> float:
    """Proxy import for solid angle utility to avoid circular imports."""
    from jlc.simulate.model_ppp import skybox_solid_angle_sr as _omega
    return float(_omega(ra_low, ra_high, dec_low, dec_high))


def compute_label_volume(cosmo, rest_wave: float, wave_min: float, wave_max: float, omega_sr: float, nz: int = 1024) -> float:
    """Compute comoving volume (Mpc^3) for a label with given rest wavelength within the band.

    V = Ω · ∫_{zmin}^{zmax} (dV/dz) dz
    where zmin = wave_min/rest − 1, zmax = wave_max/rest − 1 (clamped to >=~0).

    Notes
    - We clamp zmin to ≥1e-8 to match PPP integration behavior and avoid z=0 edge issues.
    - dV/dz is evaluated vectorially for numerical stability and performance.
    """
    if not np.isfinite(rest_wave) or rest_wave <= 0:
        return 0.0
    zmin = max(float(wave_min) / rest_wave - 1.0, 1e-8)
    zmax = max(float(wave_max) / rest_wave - 1.0, 0.0)
    if not (np.isfinite(zmin) and np.isfinite(zmax)) or zmax <= zmin:
        return 0.0
    z_grid = np.linspace(zmin, zmax, int(max(nz, 4)))
    # Use cosmology's vectorized dV/dz when available
    try:
        dVdz = np.asarray(cosmo.dV_dz(z_grid), dtype=float)
    except Exception:
        dVdz = np.array([cosmo.dV_dz(float(z)) for z in z_grid], dtype=float)
    V = float(omega_sr * np.trapz(dVdz, x=z_grid))
    return max(V, 0.0)


def luminosity_from_row(row: pd.Series, rest_wave: float, cosmo) -> float:
    """Compute CGS luminosity from a catalog row for a given line hypothesis.

    Preference order for flux columns:
    - flux_hat (measured), if present
    - flux_true (simulated truth), fallback
    Returns np.nan if invalid.
    """
    lam = float(row.get("wave_obs", np.nan))
    F = row.get("flux_hat", np.nan)
    if not np.isfinite(F):
        F = row.get("flux_true", np.nan)
    F = float(F)
    if not (np.isfinite(lam) and np.isfinite(F)) or lam <= 0 or F < 0:
        return np.nan
    z = lam / rest_wave - 1.0
    if not np.isfinite(z) or z <= 0:
        return np.nan
    dL_mpc = float(cosmo.luminosity_distance(z))
    dL_cm2 = (dL_mpc * MPC_TO_CM) ** 2
    L = 4.0 * np.pi * dL_cm2 * F
    return float(L) if np.isfinite(L) and L > 0 else np.nan


def _build_log10L_bins_from_Lstar(log10_Lstar: float, nbins: int = 20, span_decades: Tuple[float, float] = (-3.0, 3.0)) -> np.ndarray:
    lo = log10_Lstar + float(span_decades[0])
    hi = log10_Lstar + float(span_decades[1])
    return np.linspace(lo, hi, int(max(nbins, 1)) + 1)


def luminosity_from_row_true(row: pd.Series, rest_wave: float, cosmo) -> float:
    """Luminosity using flux_true."""
    lam = float(row.get("wave_obs", np.nan))
    F = float(row.get("flux_true", np.nan))
    if not (np.isfinite(lam) and np.isfinite(F)) or lam <= 0 or F <= 0:
        return np.nan
    z = lam / rest_wave - 1.0
    if not np.isfinite(z) or z <= 0:
        return np.nan
    dL_mpc = float(cosmo.luminosity_distance(z))
    dL_cm2 = (dL_mpc * MPC_TO_CM) ** 2
    L = 4.0 * np.pi * dL_cm2 * F
    return float(L) if np.isfinite(L) and L > 0 else np.nan


def luminosity_from_row_hat(row: pd.Series, rest_wave: float, cosmo) -> float:
    """Luminosity using flux_hat (measured)."""
    lam = float(row.get("wave_obs", np.nan))
    F = float(row.get("flux_hat", np.nan))
    if not (np.isfinite(lam) and np.isfinite(F)) or lam <= 0 or F <= 0:
        return np.nan
    z = lam / rest_wave - 1.0
    if not np.isfinite(z) or z <= 0:
        return np.nan
    dL_mpc = float(cosmo.luminosity_distance(z))
    dL_cm2 = (dL_mpc * MPC_TO_CM) ** 2
    L = 4.0 * np.pi * dL_cm2 * F
    return float(L) if np.isfinite(L) and L > 0 else np.nan


def effective_volume_per_L(
    cosmo,
    selection,
    rest_wave: float,
    omega_sr: float,
    wave_min: float,
    wave_max: float,
    L_vals: np.ndarray,
    nz: int = 512,
) -> np.ndarray:
    """Compute effective volume V_eff(L) by integrating Ω ∫ dV/dz · S(F(L,z), λobs(z)) dz.

    Evaluates selection at each z as completeness(F_array, scalar_lambda), looping over z for robustness.
    """
    if not (np.isfinite(omega_sr) and omega_sr > 0):
        return np.zeros_like(np.asarray(L_vals, dtype=float))
    z_grid, dVdz, lam_obs = build_zgrid_and_dVdz(cosmo, rest_wave, wave_min, wave_max, nz=nz)
    if z_grid.size == 0:
        return np.zeros_like(np.asarray(L_vals, dtype=float))
    dz = np.gradient(z_grid)
    dL = np.asarray(cosmo.luminosity_distance(z_grid), dtype=float)
    dL2 = (dL * MPC_TO_CM) ** 2
    L_vals = np.asarray(L_vals, dtype=float)
    # Precompute F(L,z)
    F_Lz = L_vals[:, None] / (4.0 * np.pi * dL2[None, :])
    # Evaluate selection column-wise over z to respect SelectionModel API
    S = np.zeros_like(F_Lz)
    for j in range(z_grid.size):
        Sj = selection.completeness(F_Lz[:, j], float(lam_obs[j]))
        # ensure shape (nL,)
        Sj = np.asarray(Sj, dtype=float).reshape(-1)
        if Sj.size != F_Lz.shape[0]:
            Sj = np.resize(Sj, F_Lz.shape[0])
        S[:, j] = np.clip(Sj, 0.0, 1.0)
    integrand = dVdz[None, :] * S
    V_eff = omega_sr * np.sum(integrand * dz[None, :], axis=1)
    # sanitize
    V_eff = np.where(np.isfinite(V_eff) & (V_eff >= 0), V_eff, 0.0)
    return V_eff


def binned_lf_simulated(
    df: pd.DataFrame,
    label: str,
    cosmo,
    selection,
    omega_sr: float,
    wave_min: float,
    wave_max: float,
    bins_log10L: np.ndarray,
    nz: int = 512,
) -> pd.DataFrame:
    """Binned LF using true flux and V_eff(L).

    Returns DataFrame with columns: label, log10L_lo, log10L_hi, phi_per_dex, err_per_dex, N, V_eff.
    """
    rest = LABEL_REST_WAVE.get(label)
    if rest is None:
        raise ValueError(f"Unknown label for LF estimation: {label}")
    sub = df[df.get("true_class", pd.Series(dtype=object)) == label]
    if sub.empty:
        return pd.DataFrame({
            "label": [], "log10L_lo": [], "log10L_hi": [], "phi_per_dex": [], "err_per_dex": [], "N": [], "V_eff": []
        })
    L = sub.apply(lambda r: luminosity_from_row_true(r, rest, cosmo), axis=1).values.astype(float)
    logL = np.log10(L, where=(L>0), out=np.full_like(L, np.nan, dtype=float))
    valid = np.isfinite(logL)
    if np.sum(valid) == 0:
        return pd.DataFrame({
            "label": [], "log10L_lo": [], "log10L_hi": [], "phi_per_dex": [], "err_per_dex": [], "N": [], "V_eff": []
        })
    counts, edges = np.histogram(logL[valid], bins=bins_log10L)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    L_centers = 10.0 ** centers
    V_eff_bins = effective_volume_per_L(cosmo, selection, rest, omega_sr, wave_min, wave_max, L_centers, nz=nz)
    V_eff_bins = np.where(V_eff_bins > 0, V_eff_bins, np.nan)
    phi = counts / (V_eff_bins * widths)
    err = np.sqrt(np.maximum(counts, 0.0)) / (V_eff_bins * widths)
    out = pd.DataFrame({
        "label": label,
        "log10L_lo": edges[:-1],
        "log10L_hi": edges[1:],
        "phi_per_dex": phi,
        "err_per_dex": err,
        "N": counts.astype(float),
        "V_eff": V_eff_bins,
    })
    return out


def binned_lf_inferred(
    df_classified: pd.DataFrame,
    label: str,
    cosmo,
    selection,
    omega_sr: float,
    wave_min: float,
    wave_max: float,
    bins_log10L: np.ndarray,
    nz: int = 512,
    weight_column: str | None = None,
    use_hard: bool = False,
) -> pd.DataFrame:
    """Inferred LF using measured flux and V_eff(L).

    Weights are p_<label> by default; can override via weight_column. If use_hard=True,
    build hard labels via argmax over available p_* columns and use weights ∈ {0,1} for the
    requested label.
    Returns DataFrame with columns: label, log10L_lo, log10L_hi, phi_per_dex, err_per_dex, N, V_eff.
    """
    rest = LABEL_REST_WAVE.get(label)
    if rest is None:
        raise ValueError(f"Unknown label for LF estimation: {label}")

    # Compute luminosity from measured flux_hat
    L = df_classified.apply(lambda r: luminosity_from_row_hat(r, rest, cosmo), axis=1).values.astype(float)
    logL = np.log10(L, where=(L>0), out=np.full_like(L, np.nan, dtype=float))

    # Build weights
    if use_hard:
        # Identify probability columns p_*
        pcols = [c for c in df_classified.columns if isinstance(c, str) and c.startswith("p_")]
        if len(pcols) == 0:
            # No probabilities to harden
            return pd.DataFrame({
                "label": [], "log10L_lo": [], "log10L_hi": [], "phi_per_dex": [], "err_per_dex": [], "N": [], "V_eff": []
            })
        P = df_classified[pcols].to_numpy(dtype=float)
        # argmax per row, map to label names without 'p_'
        idx_max = np.argmax(P, axis=1)
        hard_labels = np.array([pcols[i][2:] for i in idx_max], dtype=object)
        w = (hard_labels == label).astype(float)
    else:
        pcol = weight_column or f"p_{label}"
        if pcol not in df_classified.columns:
            return pd.DataFrame({
                "label": [], "log10L_lo": [], "log10L_hi": [], "phi_per_dex": [], "err_per_dex": [], "N": [], "V_eff": []
            })
        w = df_classified[pcol].values.astype(float)

    valid = np.isfinite(logL) & np.isfinite(w) & (w >= 0)
    if np.sum(valid) == 0:
        return pd.DataFrame({
            "label": [], "log10L_lo": [], "log10L_hi": [], "phi_per_dex": [], "err_per_dex": [], "N": [], "V_eff": []
        })

    # Weighted histogram by bin
    counts, edges = np.histogram(logL[valid], bins=bins_log10L, weights=w[valid])
    w2, _ = np.histogram(logL[valid], bins=bins_log10L, weights=(w[valid] ** 2))
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    L_centers = 10.0 ** centers
    V_eff_bins = effective_volume_per_L(cosmo, selection, rest, omega_sr, wave_min, wave_max, L_centers, nz=nz)
    V_eff_bins = np.where(V_eff_bins > 0, V_eff_bins, np.nan)
    phi = counts / (V_eff_bins * widths)
    err = np.sqrt(np.maximum(w2, 0.0)) / (V_eff_bins * widths)
    out = pd.DataFrame({
        "label": label,
        "log10L_lo": edges[:-1],
        "log10L_hi": edges[1:],
        "phi_per_dex": phi,
        "err_per_dex": err,
        "N": counts.astype(float),
        "V_eff": V_eff_bins,
    })
    return out

def default_log10L_bins_from_registry(registry, nbins: int = 20, span_decades: Tuple[float, float] = (-3.0, 3.0)) -> Dict[str, np.ndarray]:
    """Build default log10 L bins for LAE and OII based on their L* in the registry LFs."""
    out: Dict[str, np.ndarray] = {}
    try:
        lae = registry.model("lae")
        oii = registry.model("oii")
        if hasattr(lae, "lf") and hasattr(lae.lf, "log10_Lstar"):
            out["lae"] = _build_log10L_bins_from_Lstar(float(lae.lf.log10_Lstar), nbins, span_decades)
        if hasattr(oii, "lf") and hasattr(oii.lf, "log10_Lstar"):
            out["oii"] = _build_log10L_bins_from_Lstar(float(oii.lf.log10_Lstar), nbins, span_decades)
    except Exception:
        pass
    return out


def schechter_phi_per_dex(log10L: np.ndarray, lf) -> np.ndarray:
    """Compute Schechter model phi per dex for a grid of log10L given a SchechterLF.

    phi_per_dex = phi(L) * ln(10) * L, where phi(L) is per-L (CGS) from the LF.
    """
    log10L = np.asarray(log10L, dtype=float)
    L = np.power(10.0, log10L)
    # lf.phi expects L in same units as L* (CGS erg/s); returns per-L
    phi_per_L = lf.phi(L)
    phi_per_dex = phi_per_L * np.log(10.0) * L
    # sanitize
    phi_per_dex = np.where(np.isfinite(phi_per_dex) & (phi_per_dex >= 0), phi_per_dex, 0.0)
    return phi_per_dex


def _median_bin_width(bins: np.ndarray | None) -> float:
    if bins is None:
        return 0.2  # dex fallback
    bins = np.asarray(bins, dtype=float)
    if bins.ndim != 1 or bins.size < 2:
        return 0.2
    w = np.diff(bins)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.2
    return float(np.median(w))


essential_colors = {
    "binned_obs": "#1f77b4",      # blue
    "binned_inf": "#ff7f0e",      # orange
    "model": "#2ca02c",           # green
    "points": "#9467bd",          # purple
}


def plot_binned_lf(
    df_lae: Optional[pd.DataFrame],
    df_oii: Optional[pd.DataFrame],
    prefix: str,
    title: Optional[str] = None,
    registry: Optional[object] = None,
    df_inferred_points: Optional[object] = None,
    volumes: Optional[Dict[str, float]] = None,
    bins_map: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    import matplotlib.pyplot as plt
    import os

    # Enhanced plotting: overlay simulated (true-based), inferred (binned), and true model curves.
    # Accept either DataFrames or CSV file paths for simulated inputs. For inferred, accept
    # either a dict {"lae": df, "oii": df} of binned DataFrames/paths, or a single CSV/DataFrame
    # that contains both labels in a 'label' column.

    def _coerce_df(src, label_lower: str) -> pd.DataFrame:
        # Load from CSV if src is a path
        if isinstance(src, str):
            try:
                if os.path.isfile(src):
                    df = pd.read_csv(src)
                else:
                    return pd.DataFrame()
            except Exception:
                return pd.DataFrame()
        elif isinstance(src, pd.DataFrame):
            df = src.copy()
        else:
            return pd.DataFrame()
        # If a 'label' column exists, filter to desired label
        if 'label' in df.columns:
            mask = df['label'].astype(str).str.lower() == label_lower
            df = df.loc[mask].reset_index(drop=True)
        # Ensure required columns exist
        req = {"log10L_lo", "log10L_hi", "phi_per_dex"}
        if not req.issubset(set(df.columns)):
            return pd.DataFrame()
        return df

    def _extract_inferred_for(label_lower: str) -> pd.DataFrame:
        src = df_inferred_points
        if src is None:
            return pd.DataFrame()
        # If dict-like, pick by key
        try:
            if isinstance(src, dict):
                return _coerce_df(src.get(label_lower), label_lower)
        except Exception:
            pass
        # Else treat as a combined CSV/DataFrame and filter by label
        return _coerce_df(src, label_lower)

    def _plot_one(ax, df_sim_bins: pd.DataFrame, df_inf_bins: pd.DataFrame, label_name: str, lf_model) -> None:
        # Style
        ax.set_facecolor("#fafafa")
        ax.grid(True, which='both', ls=':', color='#dddddd')
        # Simulated (true-based) binned LF
        if df_sim_bins is not None and not df_sim_bins.empty:
            x_sim = 0.5 * (df_sim_bins["log10L_lo"].to_numpy() + df_sim_bins["log10L_hi"].to_numpy())
            y_sim = df_sim_bins["phi_per_dex"].to_numpy()
            yerr_sim = df_sim_bins.get("err_per_dex", pd.Series([np.nan]*len(y_sim))).to_numpy()
            ax.errorbar(
                x_sim, y_sim, yerr=yerr_sim,
                fmt='o', color=essential_colors.get("binned_obs", "#1f77b4"),
                ecolor='#6688cc', elinewidth=1.0, capsize=2.5, markersize=5,
                label=f"{label_name} simulated (binned)"
            )
        # Inferred (posterior-weighted) binned LF
        if df_inf_bins is not None and not df_inf_bins.empty:
            x_inf = 0.5 * (df_inf_bins["log10L_lo"].to_numpy() + df_inf_bins["log10L_hi"].to_numpy())
            y_inf = df_inf_bins["phi_per_dex"].to_numpy()
            yerr_inf = df_inf_bins.get("err_per_dex", pd.Series([np.nan]*len(y_inf))).to_numpy()
            ax.errorbar(
                x_inf, y_inf, yerr=yerr_inf,
                fmt='s', color=essential_colors.get("binned_inf", "#ff7f0e"),
                ecolor='#cc9966', elinewidth=1.0, capsize=2.5, markersize=5,
                label=f"{label_name} inferred (binned)"
            )
        # True model curve (Schechter per dex)
        if lf_model is not None:
            try:
                log10Lstar = float(getattr(lf_model, "log10_Lstar", np.nan))
                log10phistar = float(getattr(lf_model, "log10_phistar", np.nan))
            except Exception:
                log10Lstar = np.nan
                log10phistar = np.nan
            # Build x-limits around knee and dense grid for curve
            if np.isfinite(log10Lstar):
                xlo = log10Lstar - 2.5
                xhi = log10Lstar + 1.5
                xx = np.linspace(xlo, xhi, 300)
            else:
                # fallback bounds from data
                xs_all = []
                for dfb in [df_sim_bins, df_inf_bins]:
                    if dfb is not None and not dfb.empty:
                        xs_all.append(0.5 * (dfb["log10L_lo"].to_numpy() + dfb["log10L_hi"].to_numpy()))
                if xs_all:
                    lo = float(np.nanmin(np.concatenate(xs_all)))
                    hi = float(np.nanmax(np.concatenate(xs_all)))
                else:
                    lo, hi = 40.0, 44.0
                xx = np.linspace(lo, hi, 300)
                xlo, xhi = lo, hi
            yy = schechter_phi_per_dex(xx, lf_model)
            ax.plot(xx, yy, color=essential_colors.get("model", "#2ca02c"), lw=1.8, label=f"{label_name} model")
            # Axis limits tied to knee if available
            if np.isfinite(log10phistar):
                ylo = 10 ** (log10phistar - 2.5)
                yhi = 10 ** (log10phistar + 1.5)
                ax.set_ylim(ylo, yhi)
            ax.set_xlim(xlo, xhi)
        # Labels/scales
        ax.set_yscale('log')
        ax.set_xlabel('log10 L [erg/s]')
        ax.set_ylabel('phi [1 / (Mpc^3 dex)]')

    # Prepare LAE/OII dataframes (from DataFrame or CSV path)
    lae_sim = _coerce_df(df_lae, 'lae')
    oii_sim = _coerce_df(df_oii, 'oii')
    lae_inf = _extract_inferred_for('lae')
    oii_inf = _extract_inferred_for('oii')

    # Pull models from registry if available
    lae_lf = None
    oii_lf = None
    try:
        if registry is not None:
            lae_model = registry.model("lae")
            oii_model = registry.model("oii")
            lae_lf = getattr(lae_model, "lf", None)
            oii_lf = getattr(oii_model, "lf", None)
    except Exception:
        pass

    # LAE plot
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _plot_one(ax, lae_sim, lae_inf, 'LAE', lae_lf)
    ax.set_title((title + ' — LAE') if title else 'LAE Luminosity Function')
    ax.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(f"{prefix}_lae.png", dpi=160)
    plt.close(fig)

    # OII plot
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _plot_one(ax, oii_sim, oii_inf, 'OII', oii_lf)
    ax.set_title((title + ' — OII') if title else 'OII Luminosity Function')
    ax.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(f"{prefix}_oii.png", dpi=160)
    plt.close(fig)
