import argparse
import sys
import numpy as np
import pandas as pd

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
from jlc.simulate.simple import SkyBox, simulate_catalog, plot_distributions
from jlc.simulate import simulate_catalog_from_model


def build_default_context_and_registry(f_lim: float | None = None, wave_min: float | None = None, wave_max: float | None = None, volume_mode: str | None = None,
                                        n_fibers: int | None = None, ifu_count: int | None = None, exposure_scale: float | None = None,
                                        search_measure_scale: float | None = None, F50: float | None = None, w: float | None = None,
                                        F50_table: dict | tuple | None = None, w_table: dict | tuple | None = None):
    # Context with simple caches
    cosmo = AstropyCosmology()
    selection = SelectionModel(f_lim=f_lim, F50=F50, w=w, F50_table=F50_table, w_table=w_table)
    caches = {
        "flux_grid": FluxGrid(),
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
    F50_table = None
    w_table = None
    if getattr(args, "F50_table", None):
        try:
            from jlc.selection.base import SelectionModel as _Sel
            F50_table = _Sel.load_table(args.F50_table)
        except Exception as e:
            print(f"[jlc.classify] Warning: failed to load F50_table from {args.F50_table}: {e}")
            F50_table = None
    if getattr(args, "w_table", None):
        try:
            from jlc.selection.base import SelectionModel as _Sel
            w_table = _Sel.load_table(args.w_table)
        except Exception as e:
            print(f"[jlc.classify] Warning: failed to load w_table from {args.w_table}: {e}")
            w_table = None
    ctx, registry = build_default_context_and_registry(F50=getattr(args, "F50", None), w=getattr(args, "w", None), F50_table=F50_table, w_table=w_table)
    engine = JaynesianEngine(registry, ctx)
    out = engine.compute_log_evidence_matrix(df)
    out = engine.normalize_posteriors(out)
    out.to_csv(args.out, index=False)
    return 0


def cmd_simulate(args) -> int:
    # Build sky
    sky = SkyBox(args.ra_low, args.ra_high, args.dec_low, args.dec_high)

    # Optional completeness tables
    F50_table = None
    w_table = None
    if getattr(args, "F50_table", None):
        try:
            from jlc.selection.base import SelectionModel as _Sel
            F50_table = _Sel.load_table(args.F50_table)
        except Exception as e:
            print(f"[jlc.simulate] Warning: failed to load F50_table from {args.F50_table}: {e}")
            F50_table = None
    if getattr(args, "w_table", None):
        try:
            from jlc.selection.base import SelectionModel as _Sel
            w_table = _Sel.load_table(args.w_table)
        except Exception as e:
            print(f"[jlc.simulate] Warning: failed to load w_table from {args.w_table}: {e}")
            w_table = None

    # Optionally build empirical fake λ-PDF cache from calibration CSV
    fake_lambda_cache = None
    # Option 1: build from calibration CSV
    if getattr(args, "fake_lambda_calib", None):
        try:
            calib_df = pd.read_csv(args.fake_lambda_calib)
            from jlc.rates.observed_space import build_fake_lambda_pdf, save_fake_lambda_cache
            fake_lambda_cache = build_fake_lambda_pdf(
                calib_df.get("wave_obs", pd.Series(dtype=float)).values,
                wave_min=args.wave_min,
                wave_max=args.wave_max,
                nbins=args.fake_lambda_nbins,
            )
            if getattr(args, "fake_lambda_cache_out", None):
                try:
                    save_fake_lambda_cache(args.fake_lambda_cache_out, fake_lambda_cache)
                    print(f"[jlc.simulate] Saved fake λ-PDF cache to {args.fake_lambda_cache_out}")
                except Exception as ee:
                    print(f"[jlc.simulate] Warning: failed to save fake λ-PDF cache to {args.fake_lambda_cache_out}: {ee}")
        except Exception as e:
            print(f"[jlc.simulate] Warning: failed to build fake λ-PDF from {args.fake_lambda_calib}: {e}")
            fake_lambda_cache = None
    # Option 2: load from a precomputed cache file if not built above
    if fake_lambda_cache is None and getattr(args, "fake_lambda_cache_in", None):
        try:
            from jlc.rates.observed_space import load_fake_lambda_cache
            fake_lambda_cache = load_fake_lambda_cache(args.fake_lambda_cache_in)
        except Exception as e:
            print(f"[jlc.simulate] Warning: failed to load fake λ-PDF cache from {args.fake_lambda_cache_in}: {e}")
            fake_lambda_cache = None

    if args.from_model:
        # Build context/registry first for model-driven PPP
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
        )
        # Attach empirical fake λ-PDF cache if available
        if fake_lambda_cache is not None:
            try:
                ctx.caches["fake_lambda_pdf"] = fake_lambda_cache
            except Exception:
                pass
        # Propagate fake rate into context config for rate-density prior use
        try:
            if isinstance(ctx.config, dict):
                ctx.config["fake_rate_per_sr_per_A"] = float(args.fake_rate)
        except Exception:
            pass
        df = simulate_catalog_from_model(
            ctx=ctx,
            registry=registry,
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
        )
    else:
        # Simple fraction-based simulator
        class_fracs = {"lae": args.lae_frac, "oii": args.oii_frac, "fake": args.fake_frac}
        df = simulate_catalog(
            n=args.n,
            sky=sky,
            f_lim=args.f_lim,
            class_fracs=class_fracs,
            wave_min=args.wave_min,
            wave_max=args.wave_max,
            flux_err=args.flux_err,
            seed=args.seed,
        )
        # Build a fresh context/registry for classification
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
        )
        if fake_lambda_cache is not None:
            try:
                ctx.caches["fake_lambda_pdf"] = fake_lambda_cache
            except Exception:
                pass

    # Save simulated catalog
    if args.out_catalog:
        df.to_csv(args.out_catalog, index=False)

    # Classify using the same context/registry (PPP path reuses them)
    engine = JaynesianEngine(registry, ctx)
    out = engine.compute_log_evidence_matrix(df)
    # Use PPP expected counts as priors if available
    log_prior_weights = None
    if isinstance(getattr(ctx, "config", None), dict) and "ppp_expected_counts" in ctx.config:
        mu = ctx.config["ppp_expected_counts"]
        # Ensure only known labels and positive counts contribute
        log_prior_weights = {L: (np.log(max(mu.get(L, 0.0), 1e-300))) for L in registry.labels}
    out = engine.normalize_posteriors(out, log_prior_weights=log_prior_weights)
    if args.out_classified:
        out.to_csv(args.out_classified, index=False)

    # Optionally plot
    if args.plot_prefix:
        plot_distributions(df, args.plot_prefix)
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
    p_sim.add_argument("--flux-err", dest="flux_err", type=float, default=5e-18, help="Per-object flux error")
    p_sim.add_argument("--lae-frac", dest="lae_frac", type=float, default=0.3)
    p_sim.add_argument("--oii-frac", dest="oii_frac", type=float, default=0.3)
    p_sim.add_argument("--fake-frac", dest="fake_frac", type=float, default=0.4)
    p_sim.add_argument("--from-model", dest="from_model", action="store_true", help="Use model-driven PPP simulation instead of simple fractions")
    p_sim.add_argument("--fake-rate", dest="fake_rate", type=float, default=0.0, help="Fake rate density per sr per Angstrom for PPP mode")
    p_sim.add_argument("--nz", dest="nz", type=int, default=256, help="Number of redshift grid points for PPP mode")
    p_sim.add_argument("--volume-mode", dest="volume_mode", choices=["real", "virtual"], default="real", help="Volume mode: real (default) or virtual (no physical sources)")
    p_sim.add_argument("--fake-lambda-calib", dest="fake_lambda_calib", default=None, help="CSV with wave_obs from virtual detections to calibrate fake λ-PDF")
    p_sim.add_argument("--fake-lambda-nbins", dest="fake_lambda_nbins", type=int, default=200, help="Number of bins for fake λ-PDF calibration")
    p_sim.add_argument("--fake-lambda-cache-in", dest="fake_lambda_cache_in", default=None, help="Path to load a precomputed fake λ-PDF cache (.npz)")
    p_sim.add_argument("--fake-lambda-cache-out", dest="fake_lambda_cache_out", default=None, help="If provided and a cache is built, save it to this path (.npz)")
    # effective_search_measure knobs
    p_sim.add_argument("--n-fibers", dest="n_fibers", type=int, default=None, help="Number of fibers contributing to search (multiplier)")
    p_sim.add_argument("--ifu-count", dest="ifu_count", type=int, default=None, help="Number of IFUs contributing (multiplier)")
    p_sim.add_argument("--exposure-scale", dest="exposure_scale", type=float, default=None, help="Relative exposure/time scale (multiplier)")
    p_sim.add_argument("--search-measure-scale", dest="search_measure_scale", type=float, default=None, help="Additional scalar multiplier for effective_search_measure")
    p_sim.add_argument("--seed", dest="seed", type=int, default=12345)
    p_sim.add_argument("--out-catalog", dest="out_catalog", default="sim_catalog.csv")
    p_sim.add_argument("--out-classified", dest="out_classified", default="sim_classified.csv")
    p_sim.add_argument("--plot-prefix", dest="plot_prefix", default=None, help="If set, save distribution plots with this prefix")
    p_sim.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
