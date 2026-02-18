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
from jlc.cosmology.lookup import SimpleCosmology
from jlc.selection.base import SelectionModel
from jlc.measurements.flux import FluxMeasurement


def build_default_context_and_registry():
    # Context with simple caches
    cosmo = SimpleCosmology()
    selection = SelectionModel()
    caches = {
        "flux_grid": FluxGrid(),
    }
    ctx = SharedContext(cosmo=cosmo, selection=selection, caches=caches, config={})

    # Default Schechter parameters (placeholder values)
    lae_lf = SchechterLF(log10_Lstar=42.5, alpha=-1.5, log10_phistar=-3.0)
    oii_lf = SchechterLF(log10_Lstar=41.5, alpha=-1.3, log10_phistar=-2.5)

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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jlc", description="Jaynesian Line Classifier")
    sub = parser.add_subparsers(dest="command")

    p_classify = sub.add_parser("classify", help="Classify a catalog CSV")
    p_classify.add_argument("input", help="Input CSV path")
    p_classify.add_argument("--out", required=True, help="Output CSV path")
    p_classify.set_defaults(func=cmd_classify)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
