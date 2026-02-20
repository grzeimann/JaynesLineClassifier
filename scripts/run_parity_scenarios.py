#!/usr/bin/env python3
"""
Batch runner for PPP parity scenarios.

This helper invokes the `jlc simulate` parity harness across a few representative
configurations and writes JSON summary reports. It is intended for local use to
validate the refactored, engine‑aligned simulator against the legacy PPP implementation.

Usage:
  python scripts/run_parity_scenarios.py --out-dir parity_reports

You can tweak parameters via CLI flags; see --help for details.

Notes:
- This script does not require any external data. It uses internal defaults.
- It writes one JSON report per scenario under the output directory.
- Review the per‑scenario rtol/atol settings and adjust if needed.
"""
from __future__ import annotations
import os
import sys
import json
import argparse

from jlc.cli.main import main as jlc_main
from jlc.utils.logging import log


def _run(argv: list[str]) -> int:
    """Invoke jlc CLI with provided argv list and return exit code."""
    try:
        return jlc_main(argv)
    except SystemExit as e:
        # jlc.main raises SystemExit with code
        return int(e.code) if hasattr(e, "code") and e.code is not None else 0


def scenario_args_common(args: argparse.Namespace) -> list[str]:
    out = [
        "simulate",
        "--from-model",
        "--ra-low", str(args.ra_low), "--ra-high", str(args.ra_high),
        "--dec-low", str(args.dec_low), "--dec-high", str(args.dec_high),
        "--wave-min", str(args.wave_min), "--wave-max", str(args.wave_max),
        "--flux-err", str(args.flux_err),
        "--fake-rate", str(args.fake_rate),
        "--nz", str(args.nz),
        "--seed", str(args.seed),
        "--ppp-parity-check",
        "--parity-rtol", str(args.parity_rtol),
        "--parity-atol", str(args.parity_atol),
    ]
    return out


essential_scenarios = {
    # name: lambda args -> extra argv
    "hard_selection_real": lambda a: ["--f-lim", str(a.f_lim)],
    "smooth_selection_real": lambda a: [
        "--F50", str(a.F50), "--w", str(a.w)
    ],
    "virtual_fake_only": lambda a: [
        "--volume-mode", "virtual", "--fake-rate", str(a.fake_rate_virtual)
    ],
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run PPP parity scenarios and write JSON reports")
    p.add_argument("--out-dir", default="parity_reports", help="Directory to write parity JSON reports")
    # Sky/band defaults (modest area/band for quick runs)
    p.add_argument("--ra-low", type=float, default=150.0)
    p.add_argument("--ra-high", type=float, default=151.0)
    p.add_argument("--dec-low", type=float, default=0.0)
    p.add_argument("--dec-high", type=float, default=1.0)
    p.add_argument("--wave-min", type=float, default=5000.0)
    p.add_argument("--wave-max", type=float, default=8000.0)
    # Measurement and PPP controls
    p.add_argument("--flux-err", type=float, default=5e-18)
    p.add_argument("--nz", type=int, default=256)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--fake-rate", type=float, default=1e3)
    p.add_argument("--fake-rate-virtual", dest="fake_rate_virtual", type=float, default=2e3)
    # Selection knobs
    p.add_argument("--f-lim", type=float, default=2e-17)
    p.add_argument("--F50", type=float, default=2e-17)
    p.add_argument("--w", type=float, default=5e-18)
    # Parity tolerances
    p.add_argument("--parity-rtol", type=float, default=1e-3)
    p.add_argument("--parity-atol", type=float, default=0.0)

    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    os.makedirs(args.out_dir, exist_ok=True)

    overall_ok = True
    summary_index: dict[str, dict] = {}

    for name, builder in essential_scenarios.items():
        report_path = os.path.join(args.out_dir, f"{name}.json")
        cli_argv = scenario_args_common(args) + builder(args) + ["--parity-report", report_path, "--out-catalog", os.path.join(args.out_dir, f"{name}_catalog.csv"), "--out-classified", os.path.join(args.out_dir, f"{name}_classified.csv")]
        log(f"[parity] Running scenario: {name}\n  jlc {' '.join(cli_argv)}")
        code = _run(cli_argv)
        ok = (code == 0)
        overall_ok = overall_ok and ok
        # Read the report if present
        data = None
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                log(f"[parity] Warning: failed to load report {report_path}: {e}")
        else:
            log(f"[parity] Warning: report not found: {report_path}")
        summary_index[name] = {
            "exit_code": code,
            "report": data,
            "report_path": report_path,
        }

    index_path = os.path.join(args.out_dir, "index.json")
    try:
        with open(index_path, "w") as f:
            json.dump(summary_index, f, indent=2, sort_keys=True)
        log(f"[parity] Wrote summary index to {index_path}")
    except Exception as e:
        log(f"[parity] Warning: failed to write index.json: {e}")

    log(f"[parity] Overall status: {'OK' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
