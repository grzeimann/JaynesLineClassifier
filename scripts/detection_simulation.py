#!/usr/bin/env python3
"""
Detection Simulation Mapping Experiment

Implements DETECTION_SIMULATION_PLAN.md:
- For a set of intrinsic line profiles, map F_true to:
    Signal (detection kernel scalar), F_fit (Gaussian fit draw), F_err (per-object flux error)
- Uses the experimental kernel API in jlc.simulate.kernel.
- Assumes a constant per-object flux uncertainty sigma_F (noise level), not a full spectrum model.

Adds an optional comparison path (legacy detection) built from the original
hodgepodge code fragments you provided. Enable it with --legacy-compare to
plot side-by-side against the kernel_draw method.

Outputs
- CSV summary with mean/std over noise realizations per F_true and profile
- Diagnostic plots:
    * <prefix>_signal.png (Signal vs F_true)
    * <prefix>_fluxfit.png (F_fit vs F_true with 1σ bands)
    * <prefix>_bias.png (fractional bias (F_fit - F_true)/F_true)
    * <prefix>_ferr.png (F_err vs F_true)

This script is diagnostic and self-contained; it does not depend on a FITS cube.
"""

import numpy as np

import argparse
import os
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional imports for legacy comparison path
ASTROPY_STACK_AVAILABLE = True
try:
    from astropy.convolution import Gaussian1DKernel
    from astropy.stats import mad_std
    from scipy.interpolate import interp1d
    from astropy.modeling.models import Gaussian1D
    from astropy.modeling.fitting import LevMarLSQFitter
except Exception:
    ASTROPY_STACK_AVAILABLE = False

from jlc.simulate.kernel import (
    KernelEnv,
    GaussianLineProfile,
    SkewGaussianLineProfile,
    OIIDoubletProfile,
    draw_signal_and_flux as kernel_draw,
)


def _build_profiles() -> Dict[str, object]:
    """Registry of profile objects to probe."""
    return {
        "gaussian": GaussianLineProfile(sigma_A=4.0, gain=1.0),
        "skew_gaussian": SkewGaussianLineProfile(sigma_A=2.0, gain=1.0, skew=0.2),
        "oii_doublet": OIIDoubletProfile(sigma_A=2.0, gain=1.0, sep_A=2.8, ratio=1.0, filter_sigma_A=2.0),
    }


def _manual_convolution(a: np.ndarray, kernel: np.ndarray, return_error: bool = False) -> np.ndarray:
    """Approximate the 'manual_convolution' from the legacy snippet.

    If return_error=True, propagate uncertainties in quadrature given input a
    as 1σ errors.
    """
    a = np.asarray(a, dtype=float)
    k = np.asarray(kernel, dtype=float)
    k = k / np.sum(k) if np.sum(k) != 0 else k
    if not return_error:
        return np.convolve(a, k, mode="same")
    # Error propagation: sqrt( sum_i (a_i^2 * k_i^2) ) per output pixel, using same alignment
    # Implement with a sliding window matrix via stride convolution approximation
    # Use FFT approach for speed: variance conv equals conv of variance with k^2
    var = np.convolve(a * a, k * k, mode="same")
    var = np.clip(var, 0.0, np.inf)
    return np.sqrt(var)


def _legacy_detection_draw(F_true: float, lam0: float, sigma_F: float, rng: np.random.Generator, profile: object) -> Tuple[float, float, float]: 
    """Reworked legacy detection estimator based on the original code fragment.

    Builds a toy spectrum with constant per-pixel error, injects a Gaussian line
    of total flux F_true at lam0, smooths with a Gaussian detection kernel,
    estimates S/N curve, finds peak, fits a Gaussian, and integrates to get F_fit.
    Returns (signal≈S/N_peak, F_fit, F_err_mean_window).

    Requires optional astropy/scipy stack; if unavailable, falls back to a
    simplified draw consistent with the kernel (Gaussian draw around F_true).
    """
    if not ASTROPY_STACK_AVAILABLE:
        # Fallback: behave like kernel draw without profile gain
        F_fit = float(F_true) + float(rng.normal(0.0, sigma_F))
        return float(max(F_true, 0.0)), float(F_fit), float(abs(sigma_F))

    # Grid and kernel
    wave = np.linspace(lam0 - 24.0, lam0 + 26.0, 25)  # replicate ~3470..5540 when lam0~4505
    FWHM = 5.8
    sigma_pix = FWHM / 2.355 / 2.0  # approximate pixel sigma used in original
    Gc = Gaussian1DKernel(sigma_pix)

    # Default sigma for Gaussian fit windowing later
    line_sigma = 5.4 / 2.35

    # Build spectrum from provided profile's evaluate_profile() so that
    # the integrated area under spec_flux equals F_true. Keep per-pixel error
    # set by sigma_F.
    spec_error = np.full_like(wave, float(sigma_F))
    try:
        if hasattr(profile, "evaluate_profile") and callable(getattr(profile, "evaluate_profile")):
            shape = np.asarray(profile.evaluate_profile(wave, float(lam0)), dtype=float)
            # Ensure finite values and correct normalization (sum(shape*dlambda)=1)
            if shape.size == wave.size and np.all(np.isfinite(shape)):
                dl = np.gradient(wave)
                norm = float(np.nansum(shape * dl))
                if np.isfinite(norm) and norm > 0:
                    shape = shape / norm
                spec_flux = float(F_true) * shape
            else:
                raise ValueError("Invalid profile shape output")
        else:
            raise AttributeError("profile.evaluate_profile is not available")
    except Exception:
        # Fallback: simple Gaussian injection matching total flux F_true
        line_sigma = 5.4 / 2.35
        spec_flux = np.zeros_like(wave)
        Gline = Gaussian1D(amplitude=1.0, mean=lam0, stddev=line_sigma)
        Gline.amplitude = float(F_true) / float(line_sigma * np.sqrt(2.0 * np.pi))
        spec_flux = Gline(wave)

    # Convolve flux and error with detection kernel
    WS = _manual_convolution(spec_flux, Gc.array, return_error=False)
    WE = _manual_convolution(spec_error, Gc.array, return_error=True)

    # Robust scale calibration using central region MAD, similar to snippet
    try:
        ratio = mad_std((WS / WE)[25:-25], ignore_nan=True)
        if np.isfinite(ratio) and ratio > 0:
            WE = WE * ratio
    except Exception:
        pass

    # Interpolate S/N and get peak near lam0
    try:
        I = interp1d(wave[25:-25], (WS / WE)[25:-25], kind="quadratic", bounds_error=False, fill_value="extrapolate")
        xn = np.linspace(lam0 - 24.0, lam0 + 26.0, 76)
        sn_vals = I(xn)
        idx = int(np.nanargmax(sn_vals))
        wave_det = float(xn[idx])
        SN_det = float(sn_vals[idx])
        ind = np.argmin(np.abs(wave - wave_det))
        Sig_det = float(WS[ind])
    except Exception:
        wave_det = float(lam0)
        ind = np.argmin(np.abs(wave - wave_det))
        SN_det = float(np.nanmax(WS / np.maximum(WE, 1e-30)))
        Sig_det = float(WS[ind])


    # Gaussian fit in a ±20 Å window
    try:
        fitter = LevMarLSQFitter()
        G = Gaussian1D(mean=wave_det, stddev=line_sigma)
        G.stddev.bounds = (5.0 / 2.35, 15.0 / 2.35)
        G.mean.bounds = (wave_det - 2.0, wave_det + 2.0)
        wsel = (np.abs(wave - wave_det) < 20.0) & np.isfinite(spec_flux)
        fit = fitter(G, wave[wsel], spec_flux[wsel])
        flux_fit = float(np.trapz(fit(wave[wsel]), wave[wsel]))
    except Exception:
        # Fallback integrate original injected profile around window
        mask = (np.abs(wave - lam0) < 20.0)
        flux_fit = float(np.trapz(spec_flux[mask], wave[mask]))

    # Estimate per-object flux error as mean smoothed error in window
    try:
        wsel2 = (np.abs(wave - wave_det) < 20.0)
        F_err_obj = float(np.nanmean(WE[wsel2])) if np.any(wsel2) else float(np.nanmean(WE))
    except Exception:
        F_err_obj = float(abs(sigma_F))

    return float(Sig_det), float(flux_fit), float(F_err_obj)


def _simulate_profile(
    name: str,
    profile: object,
    F_true_grid: np.ndarray,
    lam0: float,
    sigma_F: float,
    n_realizations: int,
    rng: np.random.Generator,
    *,
    legacy_compare: bool = False,
) -> pd.DataFrame:
    """Simulate draws for one profile across F_true_grid and realizations.

    Returns a DataFrame with columns: method, profile, F_true, signal_mean, signal_std,
    F_fit_mean, F_fit_std, F_err_mean, F_err_std, frac_bias.
    """
    env = KernelEnv(lam=float(lam0), noise=float(sigma_F))
    rows: List[dict] = []
    for F in np.asarray(F_true_grid, dtype=float):
        # Kernel path
        sigs = np.empty(n_realizations, dtype=float)
        Ffits = np.empty(n_realizations, dtype=float)
        Ferrs = np.empty(n_realizations, dtype=float)
        for i in range(n_realizations):
            sig, F_fit, F_err = kernel_draw(F_true=float(F), lam=float(lam0), noise_env=env, rng=rng, profile=profile)
            sigs[i] = sig
            Ffits[i] = F_fit
            Ferrs[i] = F_err
        sm = float(np.nanmean(sigs)); ss = float(np.nanstd(sigs, ddof=1)) if n_realizations > 1 else 0.0
        fm = float(np.nanmean(Ffits)); fs = float(np.nanstd(Ffits, ddof=1)) if n_realizations > 1 else 0.0
        em = float(np.nanmean(Ferrs)); es = float(np.nanstd(Ferrs, ddof=1)) if n_realizations > 1 else 0.0
        bias = (fm - F) / F if (np.isfinite(F) and F > 0) else np.nan
        rows.append({
            "method": "kernel",
            "profile": name,
            "F_true": float(F),
            "signal_mean": sm,
            "signal_std": ss,
            "F_fit_mean": fm,
            "F_fit_std": fs,
            "F_err_mean": em,
            "F_err_std": es,
            "frac_bias": float(bias),
        })
        # Optional legacy path per F_true
        if legacy_compare:
            lsigs = np.empty(n_realizations, dtype=float)
            lfits = np.empty(n_realizations, dtype=float)
            lerrs = np.empty(n_realizations, dtype=float)
            for i in range(n_realizations):
                s2, f2, e2 = _legacy_detection_draw(float(F), float(lam0), float(sigma_F), rng, profile)
                lsigs[i] = s2; lfits[i] = f2; lerrs[i] = e2
            sm2 = float(np.nanmean(lsigs)); ss2 = float(np.nanstd(lsigs, ddof=1)) if n_realizations > 1 else 0.0
            fm2 = float(np.nanmean(lfits)); fs2 = float(np.nanstd(lfits, ddof=1)) if n_realizations > 1 else 0.0
            em2 = float(np.nanmean(lerrs)); es2 = float(np.nanstd(lerrs, ddof=1)) if n_realizations > 1 else 0.0
            bias2 = (fm2 - F) / F if (np.isfinite(F) and F > 0) else np.nan
            rows.append({
                "method": "legacy",
                "profile": name,
                "F_true": float(F),
                "signal_mean": sm2,
                "signal_std": ss2,
                "F_fit_mean": fm2,
                "F_fit_std": fs2,
                "F_err_mean": em2,
                "F_err_std": es2,
                "frac_bias": float(bias2),
            })
    return pd.DataFrame(rows)


def _plot_results(df: pd.DataFrame, prefix: str) -> None:
    if df is None or df.empty:
        return
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    profs = sorted(df["profile"].unique())

    # 1) Signal vs F_true
    plt.figure(figsize=(6, 4))
    for pname in profs:
        d = df[df.profile == pname]
        for meth, style in (("kernel", "-"), ("legacy", "--")):
            dd = d[d.method == meth] if "method" in d.columns else d
            lbl = f"{pname} ({meth})" if "method" in d.columns else pname
            plt.plot(dd.F_true, dd.signal_mean, style, label=lbl)
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("F_true [flux]")
    plt.ylabel("Signal (mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_signal.png", dpi=150)
    plt.close()

    # 2) F_fit vs F_true with 1σ bands
    plt.figure(figsize=(6, 4))
    for pname in profs:
        d = df[df.profile == pname]
        for meth, color in (("kernel", None), ("legacy", "C3")):
            dd = d[d.method == meth] if "method" in d.columns else d
            lbl = f"{pname} {meth} mean" if "method" in d.columns else f"{pname} mean"
            plt.plot(dd.F_true, dd.F_fit_mean, label=lbl, color=color)
            if "F_fit_std" in dd:
                y1 = dd.F_fit_mean - dd.F_fit_std
                y2 = dd.F_fit_mean + dd.F_fit_std
                plt.fill_between(dd.F_true.values, y1.values, y2.values, alpha=0.2, color=color)
    plt.plot(df.F_true.unique(), df.F_true.unique(), "k--", alpha=0.6, label="1:1")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("F_true")
    plt.ylabel("F_fit (mean ±1σ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_fluxfit.png", dpi=150)
    plt.close()

    # 3) Fractional bias
    plt.figure(figsize=(6, 4))
    for pname in profs:
        d = df[df.profile == pname]
        for meth, style in (("kernel", "-"), ("legacy", "--")):
            dd = d[d.method == meth] if "method" in d.columns else d
            lbl = f"{pname} ({meth})" if "method" in d.columns else pname
            plt.plot(dd.F_true, dd.frac_bias, style, label=lbl)
    plt.xscale("log")
    plt.xlabel("F_true")
    plt.ylabel("(F_fit − F_true)/F_true")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_bias.png", dpi=150)
    plt.close()

    # 4) F_err vs F_true (should be ~constant at sigma_F)
    plt.figure(figsize=(6, 4))
    for pname in profs:
        d = df[df.profile == pname]
        for meth, style in (("kernel", "-"), ("legacy", "--")):
            dd = d[d.method == meth] if "method" in d.columns else d
            lbl = f"{pname} ({meth})" if "method" in d.columns else pname
            plt.plot(dd.F_true, dd.F_err_mean, style, label=lbl)
    plt.xscale("log")
    plt.xlabel("F_true")
    plt.ylabel("F_err (mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_ferr.png", dpi=150)
    plt.close()


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detection mapping experiment: F_true → {Signal, F_fit, F_err}")
    p.add_argument("--out-prefix", default="det_map", help="Output prefix for CSV and plots (default det_map)")
    p.add_argument("--noise-sigma", type=float, default=1e-17, help="Constant flux uncertainty σ_F for all draws")
    p.add_argument("--lambda0", type=float, default=5000.0, help="Central observed wavelength for the draws")
    p.add_argument("--fmin", type=float, default=1e-19, help="Min F_true for grid (log-spaced)")
    p.add_argument("--fmax", type=float, default=1e-15, help="Max F_true for grid (log-spaced)")
    p.add_argument("--nf", type=int, default=40, help="Number of F_true grid points (log-spaced)")
    p.add_argument("--n-realizations", type=int, default=50, help="Number of noise realizations per F_true")
    p.add_argument("--seed", type=int, default=20260226, help="RNG seed")
    p.add_argument("--legacy-compare", action="store_true", help="Also run the legacy detection path and plot side-by-side")
    args = p.parse_args(argv)

    if args.legacy_compare and not ASTROPY_STACK_AVAILABLE:
        print("[det-sim] --legacy-compare requested but astropy/scipy stack not available; proceeding without legacy path")
        args.legacy_compare = False

    rng = np.random.default_rng(int(args.seed))
    F_true_grid = np.logspace(np.log10(args.fmin), np.log10(args.fmax), int(args.nf))

    profiles = _build_profiles()
    frames: List[pd.DataFrame] = []
    for name, prof in profiles.items():
        df = _simulate_profile(name, prof, F_true_grid, float(args.lambda0), float(args.noise_sigma), int(args.n_realizations), rng, legacy_compare=bool(args.legacy_compare))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Write CSV and plots
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    csv_path = f"{args.out_prefix}.csv"
    out.to_csv(csv_path, index=False)
    print(f"[det-sim] Wrote results to {csv_path} ({len(out)} rows)")

    _plot_results(out, args.out_prefix)
    print(f"[det-sim] Wrote plots with prefix {args.out_prefix}_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
