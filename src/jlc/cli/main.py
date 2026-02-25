import argparse
import sys
import time
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
from jlc.selection import build_selection_model_from_priors
from jlc.measurements.flux import FluxMeasurement
from jlc.measurements.wavelength import WavelengthMeasurement
from jlc.simulate.simple import SkyBox, plot_distributions, plot_selection_completeness, plot_label_distribution_comparison, plot_probability_circle
from jlc.simulate.field import simulate_field as simulate_field_api
from jlc.utils.logging import log
from jlc.priors import PriorRecord, load_prior_record, save_prior_record, apply_prior_to_label
from jlc.engine_noise.noise_cube_model import NoiseCube, NoiseCubeModel

# Named defaults to avoid magic numbers sprinkled around
FLUXGRID_DEFAULT_MIN = 1e-18
FLUXGRID_DEFAULT_MAX = 1e-14
FLUXGRID_DEFAULT_N = 128

def build_default_context_and_registry(wave_min: float | None = None, wave_max: float | None = None, volume_mode: str | None = None,
                                        n_fibers: int | None = None, ifu_count: int | None = None, exposure_scale: float | None = None,
                                        search_measure_scale: float | None = None,
                                        fluxgrid_min: float | None = None, fluxgrid_max: float | None = None, fluxgrid_n: int | None = None,
                                        fluxgrid_window_sigma: float | None = None, fluxgrid_window_min_n: int | None = None):
    # Context with simple caches
    cosmo = AstropyCosmology()
    selection = SelectionModel()
    # Build a configurable FluxGrid with optional guards around selection thresholds
    # Start with defaults
    fg_Fmin = FLUXGRID_DEFAULT_MIN if fluxgrid_min is None else float(fluxgrid_min)
    fg_Fmax = FLUXGRID_DEFAULT_MAX if fluxgrid_max is None else float(fluxgrid_max)
    fg_n = FLUXGRID_DEFAULT_N if fluxgrid_n is None else int(fluxgrid_n)
    # Build FluxGrid (with optional per-row windowing)
    win_sigma = None if fluxgrid_window_sigma is None else float(fluxgrid_window_sigma)
    win_min_n = 16 if fluxgrid_window_min_n is None else int(fluxgrid_window_min_n)
    _fg = FluxGrid(Fmin=fg_Fmin, Fmax=fg_Fmax, n=fg_n, window_sigma=win_sigma, window_min_n=win_min_n)
    caches = {
        "flux_grid": _fg,
    }
    config = {}
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
    if win_sigma is not None:
        config["fluxgrid_window_sigma"] = float(win_sigma)
        config["fluxgrid_window_min_n"] = int(win_min_n)
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
    wave_meas = WavelengthMeasurement()

    measurements = [flux_meas, wave_meas]

    # Initialize labels using the refactored standardized constructor
    lae = LAELabel(
        lf=lae_lf,
        selection_model=selection,
        measurement_modules=measurements,
        cosmology=cosmo,
        flux_grid=_fg,
    )
    oii = OIILabel(
        lf=oii_lf,
        selection_model=selection,
        measurement_modules=measurements,
        cosmology=cosmo,
        flux_grid=_fg,
    )
    # Tie Fake label to selection and measurement, consistent with simulator
    fake = FakeLabel(
        selection_model=selection,
        measurement_modules=measurements,
        cosmology=cosmo,
        flux_grid=_fg,
    )

    registry = LabelRegistry([lae, oii, fake])
    return ctx, registry


def _apply_prior_record_to_runtime(record: PriorRecord, registry, ctx) -> None:
    """Apply a PriorRecord to labels, measurement modules, and optionally selection in-place.

    - population: shallow-merge into label hyperparams (accepts either flat keys
      matching dataclass fields, or under a 'population' block).
    - measurements: if present, set noise/prior hyperparams for matching modules
      by name (e.g., 'flux', 'wavelength').
    - selection: if a selection.sn block is present, build S/N selection and
      attach to ctx.selection (guarded, non-breaking).
    """
    try:
        hp = record.hyperparams or {}
    except Exception:
        hp = {}
    pop_hp = dict(hp.get("population", {})) if isinstance(hp.get("population", {}), dict) else {}
    meas_hp = dict(hp.get("measurements", {})) if isinstance(hp.get("measurements", {}), dict) else {}
    # Apply to each label model that matches
    for L in registry.labels:
        m = registry.model(L)
        # label filter
        if record.label not in (None, "all", L):
            continue
        # population hyperparams (apply only recognized fields)
        to_set = dict(pop_hp)
        if len(to_set) > 0 and hasattr(m, "set_hyperparams"):
            try:
                import dataclasses as _dc
                allowed_keys = set()
                try:
                    if hasattr(m, "hyperparam_cls") and m.hyperparam_cls is not None:
                        allowed_keys = {f.name for f in _dc.fields(m.hyperparam_cls)}
                except Exception:
                    try:
                        allowed_keys = set(m.get_hyperparams_dict().keys() or [])
                    except Exception:
                        allowed_keys = set()
                if allowed_keys:
                    to_set = {k: v for k, v in to_set.items() if k in allowed_keys}
                m.set_hyperparams(**to_set)
            except Exception:
                pass
        # measurement blocks
        try:
            for mod in getattr(m, "measurement_modules", []) or []:
                name = getattr(mod, "name", None)
                if name is None:
                    continue
                blk = meas_hp.get(name, {}) or {}
                # noise/prior params might be nested under types; we expect params at blk["noise"]["params"], blk["prior"]["params"]
                try:
                    nz = blk.get("noise", {}) or {}
                    nz_params = dict(nz.get("params", {})) if isinstance(nz.get("params", {}), dict) else dict(nz)
                except Exception:
                    nz_params = {}
                try:
                    pr = blk.get("prior", {}) or {}
                    # Extract params dict if present, otherwise treat block as flat params
                    pr_params = dict(pr.get("params", {})) if isinstance(pr.get("params", {}), dict) else dict(pr)
                except Exception:
                    pr_params = {}
                # If the block carried a type, preserve it under a reserved key for modules to inspect
                try:
                    pr_type = pr.get("type", None) if isinstance(pr, dict) else None
                    if pr_type is not None:
                        pr_params = dict(pr_params)
                        pr_params["_type"] = str(pr_type)
                except Exception:
                    pass
                if len(nz_params) > 0:
                    try:
                        mod.noise_hyperparams.update(nz_params)
                    except Exception:
                        mod.noise_hyperparams = dict(nz_params)
                if len(pr_params) > 0:
                    try:
                        mod.prior_hyperparams.update(pr_params)
                    except Exception:
                        mod.prior_hyperparams = dict(pr_params)
        except Exception:
            pass
    # Optional: apply selection S/N model from prior (guarded)
    try:
        sel_blk = (hp.get("selection", {}) if isinstance(hp, dict) else {}) or {}
        if sel_blk.get("sn"):
            # build a temp SelectionModel carrying noise_model and one SN model
            tmp_sel = build_selection_model_from_priors(record)
            if tmp_sel is not None:
                # attach noise model if provided
                if getattr(tmp_sel, "noise_model", None) is not None:
                    try:
                        ctx.selection.set_noise_model(tmp_sel.noise_model)
                    except Exception:
                        setattr(ctx.selection, "noise_model", tmp_sel.noise_model)
                # attach per-label SN model
                lname = record.label or "all"
                snm = tmp_sel.sn_model_for_label(lname)
                if snm is not None:
                    try:
                        ctx.selection.set_sn_model_for(lname, snm)
                    except Exception:
                        # Fallback: internal map may be private; try attribute
                        if hasattr(ctx.selection, "_sn_models"):
                            ctx.selection._sn_models[lname] = snm
    except Exception:
        pass
    # record path for provenance if available
    try:
        if isinstance(ctx.config, dict):
            ctx.config["loaded_prior_name"] = record.name
            ctx.config["loaded_prior_label"] = record.label
    except Exception:
        pass


def cmd_classify(args) -> int:
    df = pd.read_csv(args.input)
    ctx, registry = build_default_context_and_registry(
        fluxgrid_min=getattr(args, "fluxgrid_min", None),
        fluxgrid_max=getattr(args, "fluxgrid_max", None),
        fluxgrid_n=getattr(args, "fluxgrid_n", None),
        fluxgrid_window_sigma=getattr(args, "fluxgrid_window_sigma", None),
        fluxgrid_window_min_n=getattr(args, "fluxgrid_window_min_n", None),
    )
    # Optionally load and apply a PriorRecord
    if getattr(args, "load_prior", None):
        # Accept either a single YAML path or a directory containing multiple prior YAMLs
        lp = args.load_prior
        try:
            import os
            paths = []
            if os.path.isdir(lp):
                for fn in os.listdir(lp):
                    if fn.lower().endswith((".yaml", ".yml")):
                        paths.append(os.path.join(lp, fn))
            else:
                paths = [lp]
            if len(paths) == 0:
                log(f"[jlc.classify] Warning: no prior YAMLs found in directory {lp}")
            for pth in paths:
                try:
                    rec = load_prior_record(pth)
                    _apply_prior_record_to_runtime(rec, registry, ctx)
                    log(f"[jlc.classify] Loaded prior record '{rec.name}' (scope={rec.scope}, label={rec.label}) from {pth}")
                except Exception as e:
                    log(f"[jlc.classify] Warning: failed to load/apply prior '{pth}': {e}")
        except Exception as e:
            log(f"[jlc.classify] Warning: failed to process --load-prior '{lp}': {e}")
    # If a noise cube is provided, attach it as the selection noise model (after registry/ctx creation; before engine)
    try:
        if getattr(args, "noise_cube", None):
            cube_path = str(args.noise_cube)
            default_sigma = float(getattr(args, "noise_default_sigma", 1.0) if getattr(args, "noise_default_sigma", None) is not None else 1.0)
            cube = NoiseCube.from_fits(cube_path)
            nm = NoiseCubeModel(cube, default_sigma=default_sigma)
            try:
                ctx.selection.set_noise_model(nm)
            except Exception:
                setattr(ctx.selection, "noise_model", nm)
            try:
                if isinstance(ctx.config, dict):
                    ctx.config["noise_cube_path"] = cube_path
                    ctx.config["noise_model"] = "NoiseCubeModel"
            except Exception:
                pass
            log(f"[jlc.classify] Loaded noise cube from {cube_path} with default_sigma={default_sigma}")
    except Exception as e:
        log(f"[jlc.classify] Error: failed to load --noise-cube '{getattr(args, 'noise_cube', None)}': {e}")
        return 2

    # Apply prior toggles to context config
    try:
        if isinstance(ctx.config, dict):
            ctx.config["use_rate_priors"] = not bool(getattr(args, "evidence_only", False))
            ctx.config["use_global_priors"] = not bool(getattr(args, "no_global_priors", False))
            ctx.config["use_factorized_selection"] = bool(getattr(args, "use_factorized_selection", False))
            if getattr(args, "engine_mode", None) is not None:
                ctx.config["engine_mode"] = str(args.engine_mode)
    except Exception:
        pass
    engine = JaynesianEngine(registry, ctx)
    out = engine.compute_extra_log_likelihood_matrix(df)
    out = engine.normalize_posteriors(out)
    out.to_csv(args.out, index=False)
    # Optionally save per-label PriorRecord snapshots
    if getattr(args, "save_prior_dir", None):
        import os
        save_dir = args.save_prior_dir
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass
        for L in registry.labels:
            try:
                m = registry.model(L)
                # Build hyperparams structure: population + measurements
                pop = m.get_hyperparams_dict() if hasattr(m, "get_hyperparams_dict") else {}
                meas = {}
                for mod in getattr(m, "measurement_modules", []) or []:
                    nm = getattr(mod, "name", None)
                    if nm is None:
                        continue
                    meas[nm] = {
                        "noise": {"type": "gaussian", "params": dict(getattr(mod, "noise_hyperparams", {}) or {})},
                        "prior": {"params": dict(getattr(mod, "prior_hyperparams", {}) or {})},
                    }
                rec = PriorRecord(name=f"{L}_snapshot", scope="label", label=L,
                                  hyperparams={"population": pop, "measurements": meas},
                                  source="cli_snapshot", notes="Saved after classification run")
                path = os.path.join(save_dir, f"prior_{L}.yaml")
                save_prior_record(rec, path)
                log(f"[jlc.classify] Saved PriorRecord snapshot for {L} to {path}")
            except Exception as e:
                log(f"[jlc.classify] Warning: failed to save PriorRecord for {L}: {e}")
    return 0


def cmd_simulate(args) -> int:
    # Build sky
    sky = SkyBox(args.ra_low, args.ra_high, args.dec_low, args.dec_high)


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
    ctx, registry = build_default_context_and_registry(
        wave_min=args.wave_min,
        wave_max=args.wave_max,
        volume_mode=args.volume_mode,
        n_fibers=getattr(args, "n_fibers", None),
        ifu_count=getattr(args, "ifu_count", None),
        exposure_scale=getattr(args, "exposure_scale", None),
        search_measure_scale=getattr(args, "search_measure_scale", None),
        fluxgrid_min=getattr(args, "fluxgrid_min", None),
        fluxgrid_max=getattr(args, "fluxgrid_max", None),
        fluxgrid_n=getattr(args, "fluxgrid_n", None),
        fluxgrid_window_sigma=getattr(args, "fluxgrid_window_sigma", None),
        fluxgrid_window_min_n=getattr(args, "fluxgrid_window_min_n", None),
    )

    # Optionally load and apply a PriorRecord
    if getattr(args, "load_prior", None):
        # Accept either a single YAML path or a directory containing multiple prior YAMLs
        lp = args.load_prior
        try:
            import os
            paths = []
            if os.path.isdir(lp):
                for fn in os.listdir(lp):
                    if fn.lower().endswith((".yaml", ".yml")):
                        paths.append(os.path.join(lp, fn))
            else:
                paths = [lp]
            if len(paths) == 0:
                log(f"[jlc.simulate] Warning: no prior YAMLs found in directory {lp}")
            for pth in paths:
                try:
                    rec = load_prior_record(pth)
                    _apply_prior_record_to_runtime(rec, registry, ctx)
                    log(f"[jlc.simulate] Loaded prior record '{rec.name}' (scope={rec.scope}, label={rec.label}) from {pth}")
                except Exception as e:
                    log(f"[jlc.simulate] Warning: failed to load/apply prior '{pth}': {e}")
        except Exception as e:
            log(f"[jlc.simulate] Warning: failed to process --load-prior '{lp}': {e}")
    # Apply prior toggles and debug flag to context config
    try:
        if isinstance(ctx.config, dict):
            ctx.config["use_rate_priors"] = not bool(getattr(args, "evidence_only", False))
            ctx.config["use_global_priors"] = not bool(getattr(args, "no_global_priors", False))
            ctx.config["ppp_debug"] = bool(getattr(args, "ppp_debug", False))
            # Propagate measurement simulation knobs
            if getattr(args, "wave_err", None) is not None:
                ctx.config["wave_err_sim"] = float(args.wave_err)
            if getattr(args, "flux_err", None) is not None:
                ctx.config["flux_err_sim"] = float(args.flux_err)
    except Exception:
        pass

    # If a noise cube is provided, attach it as the selection noise model (after ctx build; before simulate)
    try:
        if getattr(args, "noise_cube", None):
            cube_path = str(args.noise_cube)
            # Determine default sigma: --noise-default-sigma > --flux-err > 1.0
            if getattr(args, "noise_default_sigma", None) is not None:
                default_sigma = float(args.noise_default_sigma)
            elif getattr(args, "flux_err", None) is not None:
                default_sigma = float(args.flux_err)
            else:
                default_sigma = 1.0
            cube = NoiseCube.from_fits(cube_path)
            nm = NoiseCubeModel(cube, default_sigma=default_sigma)
            try:
                ctx.selection.set_noise_model(nm)
            except Exception:
                setattr(ctx.selection, "noise_model", nm)
            try:
                if isinstance(ctx.config, dict):
                    ctx.config["noise_cube_path"] = cube_path
                    ctx.config["noise_model"] = "NoiseCubeModel"
            except Exception:
                pass
            log(f"[jlc.simulate] Loaded noise cube from {cube_path} with default_sigma={default_sigma}")
    except Exception as e:
        log(f"[jlc.simulate] Error: failed to load --noise-cube '{getattr(args, 'noise_cube', None)}': {e}")
        return 2

    # Ensure selection NoiseModel default_sigma follows --flux-err unless explicitly overridden later
    try:
        from jlc.selection.base import NoiseModel
        sel = getattr(ctx, "selection", None)
        if sel is not None and getattr(args, "flux_err", None) is not None:
            sigma_cli = float(args.flux_err)
            # If no noise model configured (e.g., no prior with selection block), attach one
            if getattr(sel, "noise_model", None) is None:
                try:
                    sel.set_noise_model(NoiseModel(default_sigma=sigma_cli))
                except Exception:
                    setattr(sel, "noise_model", NoiseModel(default_sigma=sigma_cli))
            else:
                # Update the default sigma to match the CLI flux_err
                try:
                    sel.noise_model.default_sigma = sigma_cli
                except Exception:
                    # Replace if immutable/problematic
                    try:
                        sel.set_noise_model(NoiseModel(default_sigma=sigma_cli))
                    except Exception:
                        setattr(sel, "noise_model", NoiseModel(default_sigma=sigma_cli))
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
    tp = time.perf_counter()
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
        fake_rate_per_sr_per_A=args.fake_rate,
        seed=args.seed,
        nz=args.nz,
        snr_min=getattr(args, "snr_min", None),
    )
    t_sim_end = time.perf_counter()
    sim_time = t_sim_end - tp


    # Save simulated catalog
    if args.out_catalog:
        df.to_csv(args.out_catalog, index=False)

    # Classify using the same context/registry (PPP path reuses them)
    # If detailed timing is requested, reset FluxGrid stats so averages reflect only this classify stage
    if bool(getattr(args, "timing_detail", False)):
        try:
            fg = getattr(getattr(ctx, "caches", {}), "get", lambda k, d=None: None)("flux_grid") if hasattr(getattr(ctx, "caches", {}), "get") else getattr(getattr(ctx, "caches", None), "flux_grid", None)
            if fg is not None and hasattr(fg, "reset_stats"):
                fg.reset_stats()
        except Exception:
            pass
    t0 = time.perf_counter()
    engine = JaynesianEngine(registry, ctx)
    t1 = time.perf_counter()
    out = engine.compute_extra_log_likelihood_matrix(df)
    t2 = time.perf_counter()
    out = engine.normalize_posteriors(out)
    t3 = time.perf_counter()
    wrote = False
    if args.out_classified:
        out.to_csv(args.out_classified, index=False)
        wrote = True
    t4 = time.perf_counter()
    try:
        n_rows = len(df) if df is not None else 0
        log(
            f"[jlc.simulate.timing] simulation={t0 - tp:.3f}s, engine_init={t1 - t0:.3f}s, compute_logZ={t2 - t1:.3f}s, normalize={t3 - t2:.3f}s, write_csv={(t4 - t3) if wrote else 0.0:.3f}s, total={t4 - t0:.3f}s for N={n_rows} rows"
        )
        # Optional timing detail: average per-row FluxGrid size actually used
        if bool(getattr(args, "timing_detail", False)):
            try:
                fg = getattr(getattr(ctx, "caches", {}), "get", lambda k, d=None: None)("flux_grid") if hasattr(getattr(ctx, "caches", {}), "get") else getattr(getattr(ctx, "caches", None), "flux_grid", None)
                if fg is not None:
                    calls = int(getattr(fg, "stats_calls", 0) or 0)
                    pts = int(getattr(fg, "stats_points_total", 0) or 0)
                    avg = (pts / calls) if calls > 0 else float("nan")
                    base_n = getattr(fg, 'n', None)
                    frac = (avg / base_n * 100.0) if (base_n and base_n > 0 and np.isfinite(avg)) else float('nan')
                    win_sigma = getattr(fg, "window_sigma", None)
                    win_min_n = getattr(fg, "window_min_n", None)
                    if np.isfinite(frac):
                        log(f"[jlc.simulate.timing] FluxGrid window k={win_sigma}, min_n={win_min_n}; avg_used_points={avg:.1f} over {calls} calls (base_n={base_n}, {frac:.1f}% of base)")
                    else:
                        log(f"[jlc.simulate.timing] FluxGrid window k={win_sigma}, min_n={win_min_n}; avg_used_points={avg:.1f} over {calls} calls (base_n={base_n})")
            except Exception:
                pass
    except Exception:
        pass

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
        # New: input vs posterior-weighted distribution comparison per label
        try:
            if out is not None and not out.empty:
                plot_label_distribution_comparison(df_input=df, df_inferred=out, prefix=args.plot_prefix)
        except Exception as e:
            log(f"[jlc.simulate] Warning: failed to plot input vs posterior-weighted comparison: {e}")
        # New: probability circle plot using posterior columns p_<label>
        try:
            if out is not None and not out.empty:
                plot_probability_circle(df_inferred=out, prefix=args.plot_prefix)
        except Exception as e:
            log(f"[jlc.simulate] Warning: failed to plot probability circle: {e}")
    # Optionally save per-label PriorRecord snapshots
    if getattr(args, "save_prior_dir", None):
        import os
        save_dir = args.save_prior_dir
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass
        for L in registry.labels:
            try:
                m = registry.model(L)
                pop = m.get_hyperparams_dict() if hasattr(m, "get_hyperparams_dict") else {}
                meas = {}
                for mod in getattr(m, "measurement_modules", []) or []:
                    nm = getattr(mod, "name", None)
                    if nm is None:
                        continue
                    meas[nm] = {
                        "noise": {"type": "gaussian", "params": dict(getattr(mod, "noise_hyperparams", {}) or {})},
                        "prior": {"params": dict(getattr(mod, "prior_hyperparams", {}) or {})},
                    }
                rec = PriorRecord(name=f"{L}_snapshot", scope="label", label=L,
                                  hyperparams={"population": pop, "measurements": meas},
                                  source="cli_snapshot", notes="Saved after simulate/classify run")
                path = os.path.join(save_dir, f"prior_{L}.yaml")
                save_prior_record(rec, path)
                log(f"[jlc.simulate] Saved PriorRecord snapshot for {L} to {path}")
            except Exception as e:
                log(f"[jlc.simulate] Warning: failed to save PriorRecord for {L}: {e}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jlc", description="Jaynesian Line Classifier")
    sub = parser.add_subparsers(dest="command")

    p_classify = sub.add_parser("classify", help="Classify a catalog CSV")
    p_classify.add_argument("input", help="Input CSV path")
    p_classify.add_argument("--out", required=True, help="Output CSV path")
    p_classify.add_argument("--evidence-only", dest="evidence_only", action="store_true", help="Disable per-row rate priors; use evidences only for posteriors")
    p_classify.add_argument("--engine-mode", dest="engine_mode", choices=["rate_only", "likelihood_only", "rate_times_likelihood"], default=None, help="Posterior mode: combine rate prior and likelihood, or isolate one component")
    # FluxGrid configuration (optional)
    p_classify.add_argument("--fluxgrid-min", dest="fluxgrid_min", type=float, default=None, help="Minimum flux for FluxGrid (default 1e-18 if unset)")
    p_classify.add_argument("--fluxgrid-max", dest="fluxgrid_max", type=float, default=None, help="Maximum flux for FluxGrid (default 1e-14 if unset)")
    p_classify.add_argument("--fluxgrid-n", dest="fluxgrid_n", type=int, default=None, help="Number of flux grid points (default 128 if unset)")
    p_classify.add_argument("--load-prior", dest="load_prior", default=None, help="Path to a PriorRecord YAML to apply before classification")
    p_classify.add_argument("--save-prior-dir", dest="save_prior_dir", default=None, help="Directory to save per-label PriorRecord snapshots after classification")
    # Factorized selection toggle
    p_classify.add_argument("--use-factorized-selection", dest="use_factorized_selection", action="store_true", help="Enable factorized per-measurement selection multiplier in rate priors")
    p_classify.add_argument("--noise-cube", dest="noise_cube", default=None, help="Path to a 3D FITS noise cube (RA,Dec,λ) to use as the noise model; enables S/N-based selection if SN models are provided")
    p_classify.add_argument("--noise-default-sigma", dest="noise_default_sigma", type=float, default=1.0, help="Fallback σ used when sky position is missing (only when --noise-cube is provided)")
    p_classify.set_defaults(func=cmd_classify)

    p_classify.add_argument("--fluxgrid-window-sigma", dest="fluxgrid_window_sigma", type=float, default=None, help="If set, restrict per-row flux integration to [F_hat ± k·σ] with k=sigma; speeds up evidence and rate integrations")
    p_classify.add_argument("--fluxgrid-window-min-n", dest="fluxgrid_window_min_n", type=int, default=16, help="Minimum number of flux grid points to keep when windowing is active")
    # Timing detail diagnostics
    p_classify.add_argument("--timing-detail", dest="timing_detail", action="store_true", help="Emit detailed timing diagnostics including average per-row FluxGrid size actually used")

    p_sim = sub.add_parser("simulate", help="Generate a mock catalog and classify")
    p_sim.add_argument("--n", type=int, default=1000, help="Number of sources to simulate")
    p_sim.add_argument("--ra-low", dest="ra_low", type=float, default=0.0)
    p_sim.add_argument("--ra-high", dest="ra_high", type=float, default=10.0)
    p_sim.add_argument("--dec-low", dest="dec_low", type=float, default=-5.0)
    p_sim.add_argument("--dec-high", dest="dec_high", type=float, default=5.0)
    p_sim.add_argument("--wave-min", dest="wave_min", type=float, default=4800.0)
    p_sim.add_argument("--wave-max", dest="wave_max", type=float, default=9800.0)
    p_sim.add_argument("--flux-err", dest="flux_err", type=float, default=5e-18, help="Per-object flux error")
    p_sim.add_argument("--wave-err", dest="wave_err", type=float, default=0.0, help="Per-object wavelength error (Å) for measurement modules and simulation hooks")
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
    p_sim.add_argument("--fluxgrid-window-sigma", dest="fluxgrid_window_sigma", type=float, default=5.0, help="Restrict per-row flux integration to [F_hat ± k·σ] with k=sigma; speeds up evidence and rate computations (default 5.0)")
    p_sim.add_argument("--fluxgrid-window-min-n", dest="fluxgrid_window_min_n", type=int, default=16, help="Minimum number of flux grid points to keep when windowing is active (default 16)")
    # Luminosity Function outputs
    p_sim.add_argument("--lf-nz", dest="lf_nz", type=int, default=2048, help="Redshift grid size for V_eff(L) integration in LF binning (default 2048)")
    p_sim.add_argument("--out-lf-observed", dest="out_lf_observed", default=None, help="CSV to write binned observed LF (LAE/OII) using true_class membership")
    p_sim.add_argument("--out-lf-inferred", dest="out_lf_inferred", default=None, help="CSV to write binned inferred LF (LAE/OII) using posterior weights")
    p_sim.add_argument("--lf-bins", dest="lf_bins", type=int, default=20, help="Number of log10 L bins (built around L* by default)")
    p_sim.add_argument("--lf-plot-prefix", dest="lf_plot_prefix", default=None, help="If set, save quick plots of the binned LFs with this prefix")
    p_sim.add_argument("--lf-inferred-hard", dest="lf_inferred_hard", action="store_true", help="Use hard (argmax) derived labels for inferred LF instead of posterior-weighted counts")
    # Posterior control toggles
    p_sim.add_argument("--evidence-only", dest="evidence_only", action="store_true", help="Disable per-row rate priors; use evidences only for posteriors")
    p_sim.add_argument("--ppp-debug", dest="ppp_debug", action="store_true", help="Enable verbose PPP diagnostics (volumes, z-ranges, selection summaries)")
    p_sim.add_argument("--load-prior", dest="load_prior", default=None, help="Path to a PriorRecord YAML to apply before simulation/classification")
    p_sim.add_argument("--save-prior-dir", dest="save_prior_dir", default=None, help="Directory to save per-label PriorRecord snapshots after simulation/classification")
    # Factorized selection toggle for simulate as well
    p_sim.add_argument("--use-factorized-selection", dest="use_factorized_selection", action="store_true", help="Enable factorized per-measurement selection multiplier in rate priors during simulate/classify run")
    # Timing detail diagnostics
    p_sim.add_argument("--timing-detail", dest="timing_detail", action="store_true", help="Emit detailed timing diagnostics including average per-row FluxGrid size actually used")
    p_sim.add_argument("--noise-cube", dest="noise_cube", default=None, help="Path to a 3D FITS noise cube (RA,Dec,λ) to use as the noise model during simulate/classify")
    p_sim.add_argument("--noise-default-sigma", dest="noise_default_sigma", type=float, default=None, help="Fallback σ when sky position is missing (overrides --flux-err for default sigma if set)")
    p_sim.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
