import argparse
import sys
import numpy as np
import pandas as pd
from jlc.utils.constants import EPS_LOG, THRESH_FACTOR_LOW, THRESH_FACTOR_HIGH, FLUX_MIN_FLOOR

from jlc.types import SharedContext
from jlc.engine.engine import JaynesianEngine
from jlc.engine.flux_grid import FluxGrid
from jlc.labels.registry import LabelRegistry
from jlc.labels.lae import LAELabel
from jlc.labels.oii import OIILabel
from jlc.labels.fake import FakeLabel
from jlc.population.schechter import SchechterLF
from jlc.cosmology.lookup import AstropyCosmology
from jlc.selection.base import SelectionModel
from jlc.measurements.flux import FluxMeasurement
from jlc.simulate.simple import SkyBox, plot_distributions, plot_selection_completeness
from jlc.simulate.field import simulate_field as simulate_field_api
from jlc.utils.logging import log
from jlc.utils.cli_helpers import load_ra_dec_factor, load_completeness_tables

# Named defaults to avoid magic numbers sprinkled around
FLUXGRID_DEFAULT_MIN = 1e-18
FLUXGRID_DEFAULT_MAX = 1e-14
FLUXGRID_DEFAULT_N = 128

def build_default_context_and_registry(f_lim: float | None = None, wave_min: float | None = None, wave_max: float | None = None, volume_mode: str | None = None,
                                        n_fibers: int | None = None, ifu_count: int | None = None, exposure_scale: float | None = None,
                                        search_measure_scale: float | None = None, F50: float | None = None, w: float | None = None,
                                        F50_table: dict | tuple | None = None, w_table: dict | tuple | None = None,
                                        fluxgrid_min: float | None = None, fluxgrid_max: float | None = None, fluxgrid_n: int | None = None,
                                        ra_dec_factor: object | None = None):
    # Context with simple caches
    cosmo = AstropyCosmology()
    selection = SelectionModel(f_lim=f_lim, F50=F50, w=w, F50_table=F50_table, w_table=w_table, ra_dec_factor=ra_dec_factor)
    # Build a configurable FluxGrid with optional guards around selection thresholds
    # Start with defaults
    fg_Fmin = FLUXGRID_DEFAULT_MIN if fluxgrid_min is None else float(fluxgrid_min)
    fg_Fmax = FLUXGRID_DEFAULT_MAX if fluxgrid_max is None else float(fluxgrid_max)
    fg_n = FLUXGRID_DEFAULT_N if fluxgrid_n is None else int(fluxgrid_n)
    # Guard to ensure grid meaningfully covers selection thresholds
    thr = None
    if F50 is not None:
        thr = float(F50)
    elif f_lim is not None:
        thr = float(f_lim)
    if thr is not None and np.isfinite(thr) and thr > 0:
        # expand grid to straddle threshold comfortably
        fg_Fmin = min(fg_Fmin, max(thr * THRESH_FACTOR_LOW, FLUX_MIN_FLOOR))
        fg_Fmax = max(fg_Fmax, thr * THRESH_FACTOR_HIGH)
    # Build FluxGrid and ensure it straddles any provided selection threshold
    _fg = FluxGrid(Fmin=fg_Fmin, Fmax=fg_Fmax, n=fg_n)
    try:
        if thr is not None and np.isfinite(thr) and thr > 0:
            _fg.ensure_threshold(thr, factor_low=THRESH_FACTOR_LOW, factor_high=THRESH_FACTOR_HIGH)
    except Exception:
        pass
    caches = {
        "flux_grid": _fg,
    }
    config = {"f_lim": f_lim}
    if F50 is not None:
        config["F50"] = float(F50)
    if w is not None:
        config["w"] = float(w)
    if wave_min is not None:
        config["wave_min"] = float(wave_min)
    if wave_max is not None:
        config["wave_max"] = float(wave_max)
    if volume_mode is not None:
        config["volume_mode"] = str(volume_mode).lower()
    # Record FluxGrid settings for traceability
    config["fluxgrid_min"] = float(fg_Fmin)
    config["fluxgrid_max"] = float(fg_Fmax)
    config["fluxgrid_n"] = int(fg_n)
    # effective_search_measure knobs
    if n_fibers is not None:
        config["n_fibers"] = int(n_fibers)
    if ifu_count is not None:
        config["ifu_count"] = int(ifu_count)
    if exposure_scale is not None:
        config["exposure_scale"] = float(exposure_scale)
    if search_measure_scale is not None:
        config["search_measure_scale"] = float(search_measure_scale)
    ctx = SharedContext(cosmo=cosmo, selection=selection, caches=caches, config=config)

    # Default Schechter parameters (placeholder values)
    # Apply conservative luminosity bounds to keep PPP rates finite.
    # Bounds are set relative to L* to avoid unit inconsistencies.
    lae_Lstar = 10 ** 42.72
    oii_Lstar = 10 ** 41.4
    lae_lf = SchechterLF(log10_Lstar=42.72, alpha=-1.75, log10_phistar=-3.20,
                         Lmin=1e-3 * lae_Lstar, Lmax=1e+3 * lae_Lstar)  # Konno et al. 2016
    oii_lf = SchechterLF(log10_Lstar=41.4, alpha=-1.2, log10_phistar=-2.4,
                         Lmin=1e-3 * oii_Lstar, Lmax=1e+3 * oii_Lstar)  # Ciardullo et al.

    flux_meas = FluxMeasurement()

    lae = LAELabel(lae_lf, selection, [flux_meas])
    oii = OIILabel(oii_lf, selection, [flux_meas])
    # Tie Fake label to selection and measurement, consistent with simulator
    fake = FakeLabel(selection_model=selection, measurement_modules=[flux_meas])

    registry = LabelRegistry([lae, oii, fake])
    return ctx, registry


def cmd_classify(args) -> int:
    df = pd.read_csv(args.input)
    # Optional completeness tables
    F50_table, w_table = load_completeness_tables(args, caller="jlc.classify")
    ra_dec_fn = load_ra_dec_factor(getattr(args, "ra_dec_factor", None))
    ctx, registry = build_default_context_and_registry(F50=getattr(args, "F50", None), w=getattr(args, "w", None), F50_table=F50_table, w_table=w_table,
                                         fluxgrid_min=getattr(args, "fluxgrid_min", None), fluxgrid_max=getattr(args, "fluxgrid_max", None), fluxgrid_n=getattr(args, "fluxgrid_n", None),
                                         ra_dec_factor=ra_dec_fn)
    # Apply prior toggles to context config
    try:
        if isinstance(ctx.config, dict):
            ctx.config["use_rate_priors"] = not bool(getattr(args, "evidence_only", False))
            ctx.config["use_global_priors"] = not bool(getattr(args, "no_global_priors", False))
            if getattr(args, "engine_mode", None) is not None:
                ctx.config["engine_mode"] = str(args.engine_mode)
    except Exception:
        pass
    engine = JaynesianEngine(registry, ctx)
    out = engine.compute_extra_log_likelihood_matrix(df)
    out = engine.normalize_posteriors(out)
    out.to_csv(args.out, index=False)
    return 0


def cmd_simulate(args) -> int:
    # Build sky
    sky = SkyBox(args.ra_low, args.ra_high, args.dec_low, args.dec_high)

    # Optional completeness tables
    F50_table, w_table = load_completeness_tables(args, caller="jlc.simulate")

    # Optional: calibrate a homogeneous fake rate ρ̂ from a virtual catalog
    if getattr(args, "calibrate_fake_rate_from", None):
        try:
            calib_df = pd.read_csv(args.calibrate_fake_rate_from)
            from jlc.rates.observed_space import calibrate_fake_rate_from_catalog
            rho_hat = calibrate_fake_rate_from_catalog(
                calib_df,
                ra_low=args.ra_low, ra_high=args.ra_high,
                dec_low=args.dec_low, dec_high=args.dec_high,
                wave_min=args.wave_min, wave_max=args.wave_max,
            )
            if np.isfinite(rho_hat) and rho_hat > 0:
                log(f"[jlc.simulate] Calibrated fake rate ρ̂ ≈ {rho_hat:.6e} (1/(sr·Å)) from {args.calibrate_fake_rate_from}; overriding --fake-rate")
                args.fake_rate = float(rho_hat)
            else:
                log("[jlc.simulate] Warning: calibrated fake rate was non-positive or invalid; keeping provided --fake-rate value")
        except Exception as e:
            log(f"[jlc.simulate] Warning: failed to calibrate fake rate from {args.calibrate_fake_rate_from}: {e}")

    # Build context/registry first for model-driven PPP
    ra_dec_fn = load_ra_dec_factor(getattr(args, "ra_dec_factor", None))
    ctx, registry = build_default_context_and_registry(
        f_lim=args.f_lim,
        wave_min=args.wave_min,
        wave_max=args.wave_max,
        volume_mode=args.volume_mode,
        n_fibers=getattr(args, "n_fibers", None),
        ifu_count=getattr(args, "ifu_count", None),
        exposure_scale=getattr(args, "exposure_scale", None),
        search_measure_scale=getattr(args, "search_measure_scale", None),
        F50=getattr(args, "F50", None),
        w=getattr(args, "w", None),
        F50_table=F50_table,
        w_table=w_table,
        fluxgrid_min=getattr(args, "fluxgrid_min", None),
        fluxgrid_max=getattr(args, "fluxgrid_max", None),
        fluxgrid_n=getattr(args, "fluxgrid_n", None),
        ra_dec_factor=ra_dec_fn,
    )

    # Apply prior toggles and debug flag to context config
    try:
        if isinstance(ctx.config, dict):
            ctx.config["use_rate_priors"] = not bool(getattr(args, "evidence_only", False))
            ctx.config["use_global_priors"] = not bool(getattr(args, "no_global_priors", False))
            ctx.config["ppp_debug"] = bool(getattr(args, "ppp_debug", False))
    except Exception:
        pass

    # Propagate fake rate into context config for rate-density prior use
    try:
        if isinstance(ctx.config, dict):
            ctx.config["fake_rate_per_sr_per_A"] = float(args.fake_rate)
            ctx.config["fake_rate_rho_used"] = float(args.fake_rate)
    except Exception:
        pass
    # Record snr_min in context for traceability
    try:
        if isinstance(ctx.config, dict) and getattr(args, "snr_min", None) is not None:
            ctx.config["snr_min"] = float(args.snr_min)
    except Exception:
        pass

    log("[jlc.simulate] Using engine-aligned simulate_field() API (new architecture source of truth)")
    df = simulate_field_api(
        registry=registry,
        ctx=ctx,
        ra_low=args.ra_low,
        ra_high=args.ra_high,
        dec_low=args.dec_low,
        dec_high=args.dec_high,
        wave_min=args.wave_min,
        wave_max=args.wave_max,
        flux_err=args.flux_err,
        f_lim=args.f_lim,
        fake_rate_per_sr_per_A=args.fake_rate,
        seed=args.seed,
        nz=args.nz,
        snr_min=getattr(args, "snr_min", None),
    )


    # Save simulated catalog
    if args.out_catalog:
        df.to_csv(args.out_catalog, index=False)

    # Classify using the same context/registry (PPP path reuses them)
    engine = JaynesianEngine(registry, ctx)
    out = engine.compute_extra_log_likelihood_matrix(df)
    # Use PPP expected counts as priors if available
    log_prior_weights = None
    if isinstance(getattr(ctx, "config", None), dict) and "ppp_expected_counts" in ctx.config:
        mu = ctx.config["ppp_expected_counts"]
        # Ensure only known labels and positive counts contribute
        log_prior_weights = {L: (np.log(max(mu.get(L, 0.0), EPS_LOG))) for L in registry.labels}
    out = engine.normalize_posteriors(out, log_prior_weights=log_prior_weights)
    if args.out_classified:
        out.to_csv(args.out_classified, index=False)

    # Optionally compute and save LFs
    if getattr(args, "out_lf_observed", None) or getattr(args, "out_lf_inferred", None) or getattr(args, "lf_plot_prefix", None):
        try:
            from jlc.simulate.lf_estimation import (
                default_log10L_bins_from_registry,
                binned_lf_simulated,
                binned_lf_inferred,
                plot_binned_lf,
                skybox_solid_angle_sr,
            )
            # Build bins for LAE/OII
            bins_map = default_log10L_bins_from_registry(registry, nbins=getattr(args, "lf_bins", 20))
            # Compute solid angle and volumes
            omega = skybox_solid_angle_sr(args.ra_low, args.ra_high, args.dec_low, args.dec_high)
            # Prefer volumes computed during PPP simulate if available
            Vmap = {}
            try:
                Vcfg = getattr(ctx, "config", {}).get("ppp_label_volumes", {})
                Vmap = {k: float(v) for k, v in Vcfg.items()}
            except Exception:
                Vmap = {}
            if "lae" not in Vmap:
                from jlc.simulate.lf_estimation import compute_label_volume
                Vmap["lae"] = compute_label_volume(ctx.cosmo, 1215.67, args.wave_min, args.wave_max, omega)
            if "oii" not in Vmap:
                from jlc.simulate.lf_estimation import compute_label_volume
                Vmap["oii"] = compute_label_volume(ctx.cosmo, 3727.0, args.wave_min, args.wave_max, omega)
            # Simulated LF (using true_class and flux_true)
            df_lae_obs = df_oii_obs = None
            lf_nz = int(getattr(args, "lf_nz", 2048))
            if getattr(args, "out_lf_observed", None):
                if "lae" in bins_map:
                    df_lae_obs = binned_lf_simulated(df, "lae", ctx.cosmo, ctx.selection, omega, args.wave_min, args.wave_max, bins_map["lae"], nz=lf_nz) 
                if "oii" in bins_map:
                    df_oii_obs = binned_lf_simulated(df, "oii", ctx.cosmo, ctx.selection, omega, args.wave_min, args.wave_max, bins_map["oii"], nz=lf_nz) 
                out_path = args.out_lf_observed
                try:
                    pd.concat([d for d in [df_lae_obs, df_oii_obs] if d is not None and not d.empty]).to_csv(out_path, index=False)
                    log(f"[jlc.simulate] Wrote simulated (true-based) LF to {out_path}")
                except Exception as e:
                    log(f"[jlc.simulate] Warning: failed to write simulated LF to {out_path}: {e}")
            # Inferred (using posterior weights and flux_hat)
            df_lae_inf = df_oii_inf = None
            if getattr(args, "out_lf_inferred", None):
                if "lae" in bins_map:
                    df_lae_inf = binned_lf_inferred(out, "lae", ctx.cosmo, ctx.selection, omega, args.wave_min, args.wave_max, bins_map["lae"], use_hard=bool(getattr(args, "lf_inferred_hard", False))) 
                if "oii" in bins_map:
                    df_oii_inf = binned_lf_inferred(out, "oii", ctx.cosmo, ctx.selection, omega, args.wave_min, args.wave_max, bins_map["oii"], use_hard=bool(getattr(args, "lf_inferred_hard", False))) 
                out_path2 = args.out_lf_inferred
                try:
                    pd.concat([d for d in [df_lae_inf, df_oii_inf] if d is not None and not d.empty]).to_csv(out_path2, index=False)
                    mode = "hard labels" if bool(getattr(args, "lf_inferred_hard", False)) else "posterior-weighted"
                    log(f"[jlc.simulate] Wrote inferred LF ({mode}) to {out_path2}")
                except Exception as e:
                    log(f"[jlc.simulate] Warning: failed to write inferred LF to {out_path2}: {e}")
            # Optional plots
            if getattr(args, "lf_plot_prefix", None):
                try:
                    # Use simulated (true-based) binned LF for primary series
                    lae_plot_df = df_lae_obs
                    oii_plot_df = df_oii_obs
                    # Provide inferred (posterior-weighted) binned LF separately for overlay
                    inferred_dict = {}
                    if df_lae_inf is not None and not df_lae_inf.empty:
                        inferred_dict["lae"] = df_lae_inf
                    if df_oii_inf is not None and not df_oii_inf.empty:
                        inferred_dict["oii"] = df_oii_inf
                    inferred_src = inferred_dict if len(inferred_dict) > 0 else None
                    plot_binned_lf(
                        lae_plot_df,
                        oii_plot_df,
                        args.lf_plot_prefix,
                        title="Luminosity Function",
                        registry=registry,
                        df_inferred_points=inferred_src,
                        volumes=Vmap,
                        bins_map=bins_map,
                    )
                    log(f"[jlc.simulate] Wrote LF plots with prefix {args.lf_plot_prefix}")
                except Exception as e:
                    log(f"[jlc.simulate] Warning: failed to plot LFs: {e}")
        except Exception as e:
            log(f"[jlc.simulate] Warning: LF estimation step failed: {e}")

    # Optionally plot
    if args.plot_prefix:
        plot_distributions(df, args.plot_prefix)
        # Also plot selection completeness over the same (λ,F) grid for verification
        try:
            plot_selection_completeness(ctx.selection, args.plot_prefix)
        except Exception as e:
            log(f"[jlc.simulate] Warning: failed to plot selection completeness: {e}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jlc", description="Jaynesian Line Classifier")
    sub = parser.add_subparsers(dest="command")

    p_classify = sub.add_parser("classify", help="Classify a catalog CSV")
    p_classify.add_argument("input", help="Input CSV path")
    p_classify.add_argument("--out", required=True, help="Output CSV path")
    # SelectionModel smooth completeness knobs (optional)
    p_classify.add_argument("--F50", dest="F50", type=float, default=None, help="Flux at 50% completeness for smooth tanh selection (overrides f_lim if set)")
    p_classify.add_argument("--w", dest="w", type=float, default=None, help="Transition width for smooth tanh selection (requires F50)")
    p_classify.add_argument("--F50-table", dest="F50_table", default=None, help="Path to F50(λ) table (.npz or .csv with bin_left,value)")
    p_classify.add_argument("--w-table", dest="w_table", default=None, help="Path to w(λ) table (.npz or .csv with bin_left,value)")
    p_classify.add_argument("--evidence-only", dest="evidence_only", action="store_true", help="Disable per-row rate priors; use evidences only for posteriors")
    p_classify.add_argument("--no-global-priors", dest="no_global_priors", action="store_true", help="Do not use global prior weights (e.g., PPP expected counts)")
    p_classify.add_argument("--engine-mode", dest="engine_mode", choices=["rate_only", "likelihood_only", "rate_times_likelihood"], default=None, help="Posterior mode: combine rate prior and likelihood, or isolate one component")
    # RA/Dec-dependent selection modulation
    p_classify.add_argument("--ra-dec-factor", dest="ra_dec_factor", default=None, help="Load RA/Dec modulation function as module:function; signature g(ra, dec, lam)->[0,1]")
    # FluxGrid configuration (optional)
    p_classify.add_argument("--fluxgrid-min", dest="fluxgrid_min", type=float, default=None, help="Minimum flux for FluxGrid (default 1e-18 if unset)")
    p_classify.add_argument("--fluxgrid-max", dest="fluxgrid_max", type=float, default=None, help="Maximum flux for FluxGrid (default 1e-14 if unset)")
    p_classify.add_argument("--fluxgrid-n", dest="fluxgrid_n", type=int, default=None, help="Number of flux grid points (default 128 if unset)")
    p_classify.set_defaults(func=cmd_classify)

    p_sim = sub.add_parser("simulate", help="Generate a mock catalog and classify")
    p_sim.add_argument("--n", type=int, default=1000, help="Number of sources to simulate")
    p_sim.add_argument("--ra-low", dest="ra_low", type=float, default=0.0)
    p_sim.add_argument("--ra-high", dest="ra_high", type=float, default=10.0)
    p_sim.add_argument("--dec-low", dest="dec_low", type=float, default=-5.0)
    p_sim.add_argument("--dec-high", dest="dec_high", type=float, default=5.0)
    p_sim.add_argument("--wave-min", dest="wave_min", type=float, default=4800.0)
    p_sim.add_argument("--wave-max", dest="wave_max", type=float, default=9800.0)
    p_sim.add_argument("--f-lim", dest="f_lim", type=float, default=1e-17, help="Flux threshold for selection (used if smooth tanh not set)")
    # SelectionModel smooth completeness knobs (optional)
    p_sim.add_argument("--F50", dest="F50", type=float, default=None, help="Flux at 50% completeness for smooth tanh selection (overrides f_lim if set)")
    p_sim.add_argument("--w", dest="w", type=float, default=None, help="Transition width for smooth tanh selection (requires F50)")
    p_sim.add_argument("--F50-table", dest="F50_table", default=None, help="Path to F50(λ) table (.npz or .csv with bin_left,value)")
    p_sim.add_argument("--w-table", dest="w_table", default=None, help="Path to w(λ) table (.npz or .csv with bin_left,value)")
    # RA/Dec-dependent selection modulation
    p_sim.add_argument("--ra-dec-factor", dest="ra_dec_factor", default=None, help="Load RA/Dec modulation function as module:function; signature g(ra, dec, lam)->[0,1]")
    p_sim.add_argument("--flux-err", dest="flux_err", type=float, default=5e-18, help="Per-object flux error")
    p_sim.add_argument("--snr-min", dest="snr_min", type=float, default=None, help="Minimum S/N (flux_hat/flux_err) to include a detection in the simulated catalog")
    p_sim.add_argument("--fake-rate", dest="fake_rate", type=float, default=0.0, help="Fake rate density per sr per Angstrom for PPP mode")
    p_sim.add_argument("--calibrate-fake-rate-from", dest="calibrate_fake_rate_from", default=None, help="CSV of virtual detections to estimate a homogeneous fake rate ρ; uses current sky box and wavelength band")
    p_sim.add_argument("--nz", dest="nz", type=int, default=256, help="Number of redshift grid points for PPP mode")
    p_sim.add_argument("--volume-mode", dest="volume_mode", choices=["real", "virtual"], default="real", help="Volume mode: real (default) or virtual (no physical sources)")
    # effective_search_measure knobs
    p_sim.add_argument("--n-fibers", dest="n_fibers", type=int, default=None, help="Number of fibers contributing to search (multiplier)")
    p_sim.add_argument("--ifu-count", dest="ifu_count", type=int, default=None, help="Number of IFUs contributing (multiplier)")
    p_sim.add_argument("--exposure-scale", dest="exposure_scale", type=float, default=None, help="Relative exposure/time scale (multiplier)")
    p_sim.add_argument("--search-measure-scale", dest="search_measure_scale", type=float, default=None, help="Additional scalar multiplier for effective_search_measure")
    p_sim.add_argument("--seed", dest="seed", type=int, default=12345)
    p_sim.add_argument("--out-catalog", dest="out_catalog", default="sim_catalog.csv")
    p_sim.add_argument("--out-classified", dest="out_classified", default="sim_classified.csv")
    p_sim.add_argument("--plot-prefix", dest="plot_prefix", default=None, help="If set, save distribution plots with this prefix")
    # FluxGrid configuration (optional; shared by simulator and engine)
    p_sim.add_argument("--fluxgrid-min", dest="fluxgrid_min", type=float, default=None, help="Minimum flux for FluxGrid (default 1e-18 if unset)")
    p_sim.add_argument("--fluxgrid-max", dest="fluxgrid_max", type=float, default=None, help="Maximum flux for FluxGrid (default 1e-14 if unset)")
    p_sim.add_argument("--fluxgrid-n", dest="fluxgrid_n", type=int, default=None, help="Number of flux grid points (default 128 if unset)")
    # Luminosity Function outputs
    p_sim.add_argument("--lf-nz", dest="lf_nz", type=int, default=2048, help="Redshift grid size for V_eff(L) integration in LF binning (default 2048)")
    p_sim.add_argument("--out-lf-observed", dest="out_lf_observed", default=None, help="CSV to write binned observed LF (LAE/OII) using true_class membership")
    p_sim.add_argument("--out-lf-inferred", dest="out_lf_inferred", default=None, help="CSV to write binned inferred LF (LAE/OII) using posterior weights")
    p_sim.add_argument("--lf-bins", dest="lf_bins", type=int, default=20, help="Number of log10 L bins (built around L* by default)")
    p_sim.add_argument("--lf-plot-prefix", dest="lf_plot_prefix", default=None, help="If set, save quick plots of the binned LFs with this prefix")
    p_sim.add_argument("--lf-inferred-hard", dest="lf_inferred_hard", action="store_true", help="Use hard (argmax) derived labels for inferred LF instead of posterior-weighted counts")
    # Posterior control toggles
    p_sim.add_argument("--evidence-only", dest="evidence_only", action="store_true", help="Disable per-row rate priors; use evidences only for posteriors")
    p_sim.add_argument("--no-global-priors", dest="no_global_priors", action="store_true", help="Do not use global prior weights (e.g., PPP expected counts)")
    p_sim.add_argument("--ppp-debug", dest="ppp_debug", action="store_true", help="Enable verbose PPP diagnostics (volumes, z-ranges, selection summaries)")
    p_sim.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
