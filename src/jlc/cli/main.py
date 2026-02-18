import argparse
import sys
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


def build_default_context_and_registry(f_lim: float | None = None):
    # Context with simple caches
    cosmo = AstropyCosmology()
    selection = SelectionModel(f_lim=f_lim)
    caches = {
        "flux_grid": FluxGrid(),
    }
    ctx = SharedContext(cosmo=cosmo, selection=selection, caches=caches, config={"f_lim": f_lim})

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
    fake = FakeLabel()

    registry = LabelRegistry([lae, oii, fake])
    return ctx, registry


def cmd_classify(args) -> int:
    df = pd.read_csv(args.input)
    ctx, registry = build_default_context_and_registry()
    engine = JaynesianEngine(registry, ctx)
    out = engine.compute_log_evidence_matrix(df)
    out = engine.normalize_posteriors(out)
    out.to_csv(args.out, index=False)
    return 0


def cmd_simulate(args) -> int:
    # Build sky
    sky = SkyBox(args.ra_low, args.ra_high, args.dec_low, args.dec_high)

    if args.from_model:
        # Build context/registry first for model-driven PPP
        ctx, registry = build_default_context_and_registry(f_lim=args.f_lim)
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

    # Save simulated catalog
    if args.out_catalog:
        df.to_csv(args.out_catalog, index=False)

    # Classify with selection model using same f_lim
    ctx, registry = build_default_context_and_registry(f_lim=args.f_lim)
    engine = JaynesianEngine(registry, ctx)
    out = engine.compute_log_evidence_matrix(df)
    out = engine.normalize_posteriors(out)
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
    p_classify.set_defaults(func=cmd_classify)

    p_sim = sub.add_parser("simulate", help="Generate a mock catalog and classify")
    p_sim.add_argument("--n", type=int, default=1000, help="Number of sources to simulate")
    p_sim.add_argument("--ra-low", dest="ra_low", type=float, default=0.0)
    p_sim.add_argument("--ra-high", dest="ra_high", type=float, default=10.0)
    p_sim.add_argument("--dec-low", dest="dec_low", type=float, default=-5.0)
    p_sim.add_argument("--dec-high", dest="dec_high", type=float, default=5.0)
    p_sim.add_argument("--wave-min", dest="wave_min", type=float, default=4800.0)
    p_sim.add_argument("--wave-max", dest="wave_max", type=float, default=9800.0)
    p_sim.add_argument("--f-lim", dest="f_lim", type=float, default=1e-17, help="Flux threshold for selection")
    p_sim.add_argument("--flux-err", dest="flux_err", type=float, default=5e-18, help="Per-object flux error")
    p_sim.add_argument("--lae-frac", dest="lae_frac", type=float, default=0.3)
    p_sim.add_argument("--oii-frac", dest="oii_frac", type=float, default=0.3)
    p_sim.add_argument("--fake-frac", dest="fake_frac", type=float, default=0.4)
    p_sim.add_argument("--from-model", dest="from_model", action="store_true", help="Use model-driven PPP simulation instead of simple fractions")
    p_sim.add_argument("--fake-rate", dest="fake_rate", type=float, default=0.0, help="Fake rate density per sr per Angstrom for PPP mode")
    p_sim.add_argument("--nz", dest="nz", type=int, default=256, help="Number of redshift grid points for PPP mode")
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
