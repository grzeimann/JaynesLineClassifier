#!/usr/bin/env python3
"""
Quick utility to plot the selection completeness C(F, lambda) for LAE as
configured in a PriorRecord YAML (configs/priors/prior_lae.yaml by default).

Usage:
  python scripts/plot_lae_completeness.py \
      --prior configs/priors/prior_lae.yaml \
      --prefix lae_completeness

This will write lae_completeness_selection.png using the existing
plot_selection_completeness() helper.
"""
from __future__ import annotations
import argparse
import sys

from jlc.priors import load_prior_record
from jlc.selection import build_selection_model_from_priors
from jlc.simulate.simple import plot_selection_completeness


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Plot LAE completeness from a PriorRecord YAML")
    p.add_argument("--prior", default="configs/priors/prior_lae.yaml", help="Path to PriorRecord YAML for LAE")
    p.add_argument("--label", default="lae", help="Label name to bind SN model under (default: lae)")
    p.add_argument("--prefix", default="lae_completeness", help="Output file prefix (default: lae_completeness)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    rec = load_prior_record(args.prior)
    # Ensure the record targets the requested label for consistent binding
    try:
        if getattr(rec, "label", None) not in (None, "all", args.label):
            # Create a shallow copy-like object with the desired label
            from dataclasses import replace
            rec = replace(rec, label=args.label)
    except Exception:
        pass

    sel = build_selection_model_from_priors(rec, label_name=args.label)
    if sel is None:
        print(f"Error: selection.sn block missing or incomplete in prior {args.prior}", file=sys.stderr)
        return 2

    # Use built-in helper to generate a 2D completeness map over (lambda, F)
    plot_selection_completeness(sel, args.prefix)
    print(f"Wrote completeness plot to {args.prefix}_selection.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
