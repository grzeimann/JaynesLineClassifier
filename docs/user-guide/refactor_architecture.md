# Refactor Architecture Overview

This page summarizes the refactored architecture of JaynesLineClassifier (JLC) and how simulation and inference are now aligned under a common generative model.

Key goals of the refactor:
- The same generative model is used for simulation and probability assignment.
- The engine is thin and declarative.
- Labels are self-contained.
- SelectionModel is standalone and shared.
- Simulation and inference are mathematically aligned.

## Components

- SelectionModel
  - Computes completeness C(F, λ[, RA, Dec]) ∈ [0,1].
  - Can be configured as a hard threshold (f_lim) or a smooth tanh transition (F50, w), optionally with wavelength tables F50(λ), w(λ).
  - Optionally accepts a RA/Dec modulation factor g(ra, dec, λ) ∈ [0,1] provided by the user.

- LabelModel (one per label: LAE, OII, Fake)
  - rate_density(row, ctx): observed-space PPP intensity λ_L(x) that includes the intrinsic LF or fake rate, cosmology, selection completeness, and effective search measure.
  - extra_log_likelihood(row, ctx): measurement-only likelihood terms not already captured by rate_density (e.g., flux marginalization using a shared FluxGrid).
  - simulate_catalog(...): per-label simulator that uses the same ingredients as rate_density for engine-aligned simulation.

- JaynesianEngine
  - Loops over rows and labels, computes:
    log P(L | x) ∝ log λ_L(x) + extra_log_likelihood_L(x) + log W_L
  - Supports modes: rate_only, likelihood_only, rate_times_likelihood (default).
  - Preferred API: compute_extra_log_likelihood_matrix(df) + normalize_posteriors(...).

- FluxGrid
  - Shared log-spaced flux grid for marginalization over true flux.
  - Configurable via CLI or context; auto-expands to straddle selection thresholds.

## Simulator alignment

The engine-aligned simulator aggregates per-label simulate_catalog() calls via the unified API:

- jlc.simulate.field.simulate_field(registry, ctx, ...)
- This is now the sole simulation backend used by the CLI.
- Expected counts and volumes computed during simulation are recorded in ctx.config:
  - ppp_expected_counts: {lae, oii, fake, total}
  - ppp_label_volumes: {lae, oii}

## PPP parity harness

To validate that the new engine-aligned simulator matches the legacy PPP implementation, the CLI provides a parity harness:

- Use jlc simulate --from-model --ppp-parity-check [--parity-rtol 1e-3 --parity-atol 0.0]
- The harness logs per-label comparisons of expected counts μ and volumes V and stores a machine-readable summary in ctx.config['ppp_parity_summary'].
- Optional: write a JSON report with --parity-report PATH.

Example:

```bash
jlc simulate --from-model \
  --ra-low 150 --ra-high 151 \
  --dec-low 0 --dec-high 1 \
  --wave-min 5000 --wave-max 8000 \
  --f-lim 2e-17 --fake-rate 1e3 \
  --ppp-parity-check --parity-rtol 1e-3 --parity-atol 0 \
  --parity-report parity.json \
  --out-catalog sim_catalog.csv --out-classified sim_classified.csv
```

Sample parity.json (truncated):

```json
{
  "rtol": 0.001,
  "atol": 0.0,
  "counts_ok": {"lae": true, "oii": true, "fake": true, "total": true},
  "vols_ok": {"lae": true, "oii": true},
  "mu_engine": {"lae": 12.34, "oii": 8.76, "fake": 5.00, "total": 26.10},
  "mu_legacy": {"lae": 12.34, "oii": 8.76, "fake": 5.00, "total": 26.10}
}
```

Interpreting results:
- All counts_ok and vols_ok flags should be True within your tolerances. Typical defaults are rtol=1e-3, atol=0.
- If parity fails:
  - Ensure the same selection configuration is used in both paths (F50/w vs f_lim, and any λ-tables).
  - Check FluxGrid bounds cover selection thresholds (the CLI auto-expands around thresholds, but manual overrides can restrict the range).
  - Verify volume_mode and nz are the same.

## Global rename (status)

The terminology has moved from log_evidence to extra_log_likelihood.

Status:
- Engine, CLI, and tests use compute_extra_log_likelihood_matrix for evidences. ✓
- LabelModel.log_evidence is retained only as a deprecated shim (DeprecationWarning, stacklevel=2) delegating to extra_log_likelihood for one minor version. ✓
- The engine’s preferred path does not fall back to log_evidence; missing values are treated as −∞. ✓
- Documentation has been updated to prefer extra_log_likelihood throughout. ✓

Migration tip:
- For custom labels, implement extra_log_likelihood(row, ctx) and optionally keep a thin log_evidence shim during the transition window.

## Best practices and tips

- When using smooth selection (F50, w), provide wavelength tables if selection varies with λ.
- Consider providing a RA/Dec modulation factor if spatial completeness variations (e.g., masks) are available.
- For Fake label priors, you can calibrate an empirical λ-PDF from virtual detections and supply it to the context via the CLI (see --fake-lambda-* options).
- The rate_only engine mode is useful for validating PPP-based expectations; likelihood_only isolates measurement information; combined is the default generative classifier.

## References

- CLI parity flags: see docs/user-guide/cli.md for --ppp-parity-check, --parity-rtol/--parity-atol, and --parity-report.
- Tests covering engine modes, selection behavior, PPP sanity, and LF regression live under tests/ (e.g., tests/test_engine_modes.py, tests/test_selection_model.py, tests/test_sanity.py, tests/test_lf_regression.py).
